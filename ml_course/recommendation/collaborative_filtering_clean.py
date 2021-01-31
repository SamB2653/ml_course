from __future__ import print_function
from urllib.request import urlretrieve
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import collections
import sklearn
import sklearn.manifold
import tensorflow.compat.v1 as tf
import altair as alt
import altair_viewer
import zipfile


def mask(df, key, function):
    """ Returns a filtered dataframe, by applying function to key """
    return df[function(df[key])]


def flatten_cols(df):
    df.columns = [' '.join(col).strip() for col in df.columns.values]
    return df


# Since some movies can belong to more than one genre, we create different 'genre' columns as follows:
# * all_genres: all the active genres of the movie
# * genre: randomly sampled from the active genres
def mark_genres(movies, genres):
    def get_random_genre(gs):
        active = [genre for genre, g in zip(genres, gs) if g == 1]
        if len(active) == 0:
            return 'Other'
        return np.random.choice(active)

    def get_all_genres(gs):
        active = [genre for genre, g in zip(genres, gs) if g == 1]
        if len(active) == 0:
            return 'Other'
        return '-'.join(active)

    movies['genre'] = [get_random_genre(gs) for gs in zip(*[movies[genre] for genre in genres])]
    movies['all_genres'] = [get_all_genres(gs) for gs in zip(*[movies[genre] for genre in genres])]


# The movies file contains a binary feature for each genre
def binary_feature(users, ratings):
    genre_cols = [
        "genre_unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
    movies_cols = ['movie_id', 'title', 'release_date', "video_release_date", "imdb_url"] + genre_cols
    movies = pd.read_csv('ml-100k/u.item', sep='|', names=movies_cols, encoding='latin-1')

    # Format some of the data, since the ids start at 1, shift them to start at 0
    users["user_id"] = users["user_id"].apply(lambda x: str(x - 1))
    movies["movie_id"] = movies["movie_id"].apply(lambda x: str(x - 1))
    movies["year"] = movies['release_date'].apply(lambda x: str(x).split('-')[-1])
    ratings["movie_id"] = ratings["movie_id"].apply(lambda x: str(x - 1))
    ratings["user_id"] = ratings["user_id"].apply(lambda x: str(x - 1))
    ratings["rating"] = ratings["rating"].apply(lambda x: float(x))

    genre_occurences = movies[genre_cols].sum().to_dict()  # Count the number of movies where a genre is assigned
    mark_genres(movies, genre_cols)
    movielens = ratings.merge(movies, on='movie_id').merge(users, on='user_id')  # all the MovieLens data

    return genre_occurences, movielens, movies


# Function to split the data into training and test sets
def split_dataframe(df, holdout_fraction=0.1):
    """Splits a DataFrame into training and test sets.
    Args:
      df: a dataframe.
      holdout_fraction: fraction of dataframe rows to use in the test set.
    Returns:
      train: dataframe for training
      test: dataframe for testing
    """
    test = df.sample(frac=holdout_fraction, replace=False)
    train = df[~df.index.isin(test.index)]
    return train, test


# A function that generates a histogram of filtered data
def filtered_hist(field, label, filter):
    """Creates a layered chart of histograms. The first layer (light gray) contains the histogram of the full data,
    and the second contains the histogram of the filtered data
    Args:
      field: the field for which to generate the histogram
      label: String label of the histogram
      filter: an alt.Selection object to be used to filter the data
    """
    base = alt.Chart().mark_bar().encode(x=alt.X(field, bin=alt.Bin(maxbins=10), title=label),
                                         y="count()", ).properties(width=300, )
    return alt.layer(
        base.transform_filter(filter),
        base.encode(color=alt.value('lightgray'), opacity=alt.value(.7)), ).resolve_scale(y='independent')


def get_movie_ratings(movies, ratings):
    movies_ratings = movies.merge(
        ratings.groupby('movie_id', as_index=False).agg({'rating': ['count', 'mean']}).flatten_cols(), on='movie_id')
    return movies_ratings


def filter_and_chart():
    genre_filter = alt.selection_multi(fields=['genre'])

    genre_chart = alt.Chart().mark_bar().encode(x="count()", y=alt.Y('genre'), color=alt.condition(
        genre_filter, alt.Color("genre:N"), alt.value('lightgray'))).properties(height=300, selection=genre_filter)
    return genre_filter, genre_chart


def explore_dataset(users, movies, ratings):
    # Exploring the MovieLens Data (Users)
    print(users.describe())  # User features
    print(users.describe(include=[np.object]))  # Categorical user features

    # Create filters to slice the data
    occupation_filter = alt.selection_multi(fields=["occupation"])
    occupation_chart = alt.Chart().mark_bar().encode(
        x="count()",
        y=alt.Y("occupation:N"),
        color=alt.condition(
            occupation_filter,
            alt.Color("occupation:N", scale=alt.Scale(scheme='category20')),
            alt.value("lightgray")),
    ).properties(width=300, height=300, selection=occupation_filter)

    # Create the chart
    users_ratings = (
        ratings.groupby('user_id', as_index=False).agg({'rating': ['count', 'mean']}).flatten_cols().merge(users,
                                                                                                           on='user_id'))

    # Create a chart for the count, and one for the mean.
    altair_viewer.show(
        alt.hconcat(filtered_hist('rating count', '# ratings / user', occupation_filter),
                    filtered_hist('rating mean', 'mean user rating', occupation_filter), occupation_chart,
                    data=users_ratings))

    # Exploring the MovieLens Data (Movies)
    movies_ratings = get_movie_ratings(movies, ratings)

    genre_filter, genre_chart = filter_and_chart()

    (movies_ratings[['title', 'rating count', 'rating mean']]
     .sort_values('rating count', ascending=False)
     .head(10))

    (movies_ratings[['title', 'rating count', 'rating mean']]
     .mask('rating count', lambda x: x > 20)
     .sort_values('rating mean', ascending=False)
     .head(10))

    # Display the number of ratings and average rating per movie
    altair_viewer.show(alt.hconcat(
        filtered_hist('rating count', '# ratings / movie', genre_filter),
        filtered_hist('rating mean', 'mean movie rating', genre_filter),
        genre_chart,
        data=movies_ratings))


def build_rating_sparse_tensor(users, movies,
                               ratings_df):  # Build a tf.SparseTensor representation of the Rating Matrix
    """
    Args:
      ratings_df: a pd.DataFrame with `user_id`, `movie_id` and `rating` columns.
    Returns:
      A tf.SparseTensor representing the ratings matrix.
    """

    indices = ratings_df[['user_id', 'movie_id']].values
    values = ratings_df['rating'].values

    return tf.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=[users.shape[0], movies.shape[0]])


def sparse_mean_square_error(sparse_ratings, user_embeddings, movie_embeddings):  # Calculating the error
    """
    Args:
      sparse_ratings: A SparseTensor rating matrix, of dense_shape [N, M]
      user_embeddings: A dense Tensor U of shape [N, k] where k is the embedding
        dimension, such that U_i is the embedding of user i.
      movie_embeddings: A dense Tensor V of shape [M, k] where k is the embedding
        dimension, such that V_j is the embedding of movie j.
    Returns:
      A scalar Tensor representing the MSE between the true ratings and the
        model's predictions.
    """
    predictions = tf.gather_nd(
        tf.matmul(user_embeddings, movie_embeddings, transpose_b=True),
        sparse_ratings.indices)
    loss = tf.losses.mean_squared_error(sparse_ratings.values, predictions)
    return loss


class CFModel(object):  # Training a Matrix Factorization model
    """Simple class that represents a collaborative filtering model"""

    def __init__(self, embedding_vars, loss, metrics=None):
        """Initializes a CFModel.
        Args:
          embedding_vars: A dictionary of tf.Variables.
          loss: A float Tensor. The loss to optimize.
          metrics: optional list of dictionaries of Tensors. The metrics in each
            dictionary will be plotted in a separate figure during training.
        """
        self._embedding_vars = embedding_vars
        self._loss = loss
        self._metrics = metrics
        self._embeddings = {k: None for k in embedding_vars}
        self._session = None

    @property
    def embeddings(self):
        """The embeddings dictionary."""
        return self._embeddings

    def train(self, num_iterations=100, learning_rate=1.0, plot_results=True,  # Changed from True
              optimizer=tf.train.GradientDescentOptimizer):
        """Trains the model.
        Args:
          iterations: number of iterations to run.
          learning_rate: optimizer learning rate.
          plot_results: whether to plot the results at the end of training.
          optimizer: the optimizer to use. Default to GradientDescentOptimizer.
        Returns:
          The metrics dictionary evaluated at the last iteration.
        """
        with self._loss.graph.as_default():
            opt = optimizer(learning_rate)
            train_op = opt.minimize(self._loss)
            local_init_op = tf.group(
                tf.variables_initializer(opt.variables()),
                tf.local_variables_initializer())
            if self._session is None:
                self._session = tf.Session()
                with self._session.as_default():
                    self._session.run(tf.global_variables_initializer())
                    self._session.run(tf.tables_initializer())
                    tf.train.start_queue_runners()

        with self._session.as_default():
            local_init_op.run()
            iterations = []
            metrics = self._metrics or ({},)
            metrics_vals = [collections.defaultdict(list) for _ in self._metrics]

            # Train and append results.
            for i in range(num_iterations + 1):
                _, results = self._session.run((train_op, metrics))
                if (i % 10 == 0) or i == num_iterations:
                    print("\r iteration %d: " % i + ", ".join(
                        ["%s=%f" % (k, v) for r in results for k, v in r.items()]),
                          end='')
                    iterations.append(i)
                    for metric_val, result in zip(metrics_vals, results):
                        for k, v in result.items():
                            metric_val[k].append(v)

            for k, v in self._embedding_vars.items():
                self._embeddings[k] = v.eval()

            if plot_results:
                # Plot the metrics.
                num_subplots = len(metrics) + 1
                fig = plt.figure()
                fig.set_size_inches(num_subplots * 10, 8)
                for i, metric_vals in enumerate(metrics_vals):
                    ax = fig.add_subplot(1, num_subplots, i + 1)
                    for k, v in metric_vals.items():
                        ax.plot(iterations, v, label=k)
                    ax.set_xlim([1, num_iterations])
                    ax.legend()
                    plt.show()
            return results


def build_model(users, movies, ratings, embedding_dim=3,
                init_stddev=1.):  # Build a Matrix Factorization model and train it
    """
    Args:
      ratings: a DataFrame of the ratings
      embedding_dim: the dimension of the embedding vectors.
      init_stddev: float, the standard deviation of the random initial embeddings.
    Returns:
      model: a CFModel.
    """
    # Split the ratings DataFrame into train and test.
    train_ratings, test_ratings = split_dataframe(ratings)
    # SparseTensor representation of the train and test datasets.
    A_train = build_rating_sparse_tensor(users, movies, train_ratings)
    A_test = build_rating_sparse_tensor(users, movies, test_ratings)
    # Initialize the embeddings using a normal distribution.
    U = tf.Variable(tf.random_normal(
        [A_train.dense_shape[0], embedding_dim], stddev=init_stddev))
    V = tf.Variable(tf.random_normal(
        [A_train.dense_shape[1], embedding_dim], stddev=init_stddev))
    train_loss = sparse_mean_square_error(A_train, U, V)
    test_loss = sparse_mean_square_error(A_test, U, V)
    metrics = {
        'train_error': train_loss,
        'test_error': test_loss
    }
    embeddings = {
        "user_id": U,
        "movie_id": V
    }
    return CFModel(embeddings, train_loss, [metrics])


def compute_scores(query_embedding, item_embeddings, measure):
    """Computes the scores of the candidates given a query.
    Args:
      query_embedding: a vector of shape [k], representing the query embedding.
      item_embeddings: a matrix of shape [N, k], such that row i is the embedding
        of item i.
      measure: a string specifying the similarity measure to be used. Can be
        either DOT or COSINE.
    Returns:
      scores: a vector of shape [N], such that scores[i] is the score of item i.
    """
    u = query_embedding
    V = item_embeddings
    if measure == 'cosine':
        V = V / np.linalg.norm(V, axis=1, keepdims=True)
        u = u / np.linalg.norm(u)
    scores = u.dot(V.T)
    return scores


# User recommendations and nearest neighbors
def user_recommendations(USER_RATINGS, movies, ratings, model, measure, exclude_rated=False, k=6):
    if USER_RATINGS:
        scores = compute_scores(model.embeddings["user_id"][943], model.embeddings["movie_id"], measure)
        score_key = measure + ' score'
        df = pd.DataFrame({
            score_key: list(scores),
            'movie_id': movies['movie_id'],
            'titles': movies['title'],
            'genres': movies['all_genres'],
        })
        if exclude_rated:
            # remove movies that are already rated
            rated_movies = ratings[ratings.user_id == "943"]["movie_id"].values
            df = df[df.movie_id.apply(lambda movie_id: movie_id not in rated_movies)]
        display.display(df.sort_values([score_key], ascending=False).head(k))


def movie_neighbors(movies, model, title_substring, measure, k=6):
    # Search for movie ids that match the given substring.
    ids = movies[movies['title'].str.contains(title_substring)].index.values
    titles = movies.iloc[ids]['title'].values
    if len(titles) == 0:
        raise ValueError("Found no movies with title %s" % title_substring)
    print("Nearest neighbors of : %s." % titles[0])
    if len(titles) > 1:
        print("[Found more than one matching movie. Other candidates: {}]".format(
            ", ".join(titles[1:])))
    movie_id = ids[0]
    scores = compute_scores(
        model.embeddings["movie_id"][movie_id], model.embeddings["movie_id"],
        measure)
    score_key = measure + ' score'
    df = pd.DataFrame({
        score_key: list(scores),
        'titles': movies['title'],
        'genres': movies['all_genres']
    })
    display.display(df.sort_values([score_key], ascending=False).head(k))


# Movie Embedding Norm
def movie_embedding_norm(movies, movies_ratings, models):
    """Visualizes the norm and number of ratings of the movie embeddings.
    Args:
      model: A MFModel object.
    """
    if not isinstance(models, list):
        models = [models]
    df = pd.DataFrame({
        'title': movies['title'],
        'genre': movies['genre'],
        'num_ratings': movies_ratings['rating count'],
    })
    charts = []
    brush = alt.selection_interval()
    for i, model in enumerate(models):
        norm_key = 'norm' + str(i)
        df[norm_key] = np.linalg.norm(model.embeddings["movie_id"], axis=1)
        nearest = alt.selection(
            type='single', encodings=['x', 'y'], on='mouseover', nearest=True,
            empty='none')
        base = alt.Chart().mark_circle().encode(
            x='num_ratings',
            y=norm_key,
            color=alt.condition(brush, alt.value('#4c78a8'), alt.value('lightgray'))
        ).properties(
            selection=nearest).add_selection(brush)
        text = alt.Chart().mark_text(align='center', dx=5, dy=-5).encode(
            x='num_ratings', y=norm_key,
            text=alt.condition(nearest, 'title', alt.value('')))
        charts.append(alt.layer(base, text))
    return altair_viewer.show(alt.hconcat(*charts, data=df))


def visualize_movie_embeddings(data, x, y, genre_filter, genre_chart):
    nearest = alt.selection(
        type='single', encodings=['x', 'y'], on='mouseover', nearest=True,
        empty='none')
    base = alt.Chart().mark_circle().encode(
        x=x,
        y=y,
        color=alt.condition(genre_filter, "genre", alt.value("whitesmoke")),
    ).properties(
        width=600,
        height=600,
        selection=nearest)
    text = alt.Chart().mark_text(align='left', dx=5, dy=-5).encode(
        x=x,
        y=y,
        text=alt.condition(nearest, 'title', alt.value('')))
    return altair_viewer.show(alt.hconcat(alt.layer(base, text), genre_chart, data=data))


def tsne_movie_embeddings(model, movies, genre_filter, genre_chart):
    """Visualizes the movie embeddings, projected using t-SNE with Cosine measure.
    Args:
      model: A MFModel object.
    """
    tsne = sklearn.manifold.TSNE(
        n_components=2, perplexity=40, metric='cosine', early_exaggeration=10.0,
        init='pca', verbose=True, n_iter=400)

    print('Running t-SNE...')
    V_proj = tsne.fit_transform(model.embeddings["movie_id"])
    movies.loc[:, 'x'] = V_proj[:, 0]
    movies.loc[:, 'y'] = V_proj[:, 1]
    return visualize_movie_embeddings(movies, 'x', 'y', genre_filter, genre_chart)


def gravity(U, V):
    """Creates a gravity loss given two embedding matrices."""
    return 1. / (U.shape[0].value * V.shape[0].value) * tf.reduce_sum(
        tf.matmul(U, U, transpose_a=True) * tf.matmul(V, V, transpose_a=True))


def build_regularized_model(users, movies,
                            ratings, embedding_dim=3, regularization_coeff=.1, gravity_coeff=1., init_stddev=0.1):
    """
    Args:
      ratings: the DataFrame of movie ratings.
      embedding_dim: The dimension of the embedding space.
      regularization_coeff: The regularization coefficient lambda.
      gravity_coeff: The gravity regularization coefficient lambda_g.
    Returns:
      A CFModel object that uses a regularized loss.
    """
    # Split the ratings DataFrame into train and test.
    train_ratings, test_ratings = split_dataframe(ratings)
    # SparseTensor representation of the train and test datasets.
    A_train = build_rating_sparse_tensor(users, movies, train_ratings)
    A_test = build_rating_sparse_tensor(users, movies, test_ratings)
    U = tf.Variable(tf.random_normal(
        [A_train.dense_shape[0], embedding_dim], stddev=init_stddev))
    V = tf.Variable(tf.random_normal(
        [A_train.dense_shape[1], embedding_dim], stddev=init_stddev))

    error_train = sparse_mean_square_error(A_train, U, V)
    error_test = sparse_mean_square_error(A_test, U, V)
    gravity_loss = gravity_coeff * gravity(U, V)
    regularization_loss = regularization_coeff * (
            tf.reduce_sum(U * U) / U.shape[0].value + tf.reduce_sum(V * V) / V.shape[0].value)
    total_loss = error_train + regularization_loss + gravity_loss
    losses = {
        'train_error_observed': error_train,
        'test_error_observed': error_test,
    }
    loss_components = {
        'observed_loss': error_train,
        'regularization_loss': regularization_loss,
        'gravity_loss': gravity_loss,
    }
    embeddings = {"user_id": U, "movie_id": V}

    return CFModel(embeddings, total_loss, [losses, loss_components])


def load_data():  # Load and download MovieLens data
    urlretrieve("http://files.grouplens.org/datasets/movielens/ml-100k.zip", "movielens.zip")
    zip_ref = zipfile.ZipFile('movielens.zip', "r")
    zip_ref.extractall()
    print("Data set contains:")
    print(zip_ref.read('ml-100k/u.info'))

    # Load each data set (users, movies, and ratings)
    users_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    users = pd.read_csv('ml-100k/u.user', sep='|', names=users_cols, encoding='latin-1')
    ratings_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=ratings_cols, encoding='latin-1')

    return users, ratings


def main():
    tf.disable_v2_behavior()
    tf.logging.set_verbosity(tf.logging.ERROR)
    pd.options.display.float_format = '{:.3f}'.format
    pd.options.display.max_rows = 15
    pd.options.display.width = 0
    USER_RATINGS = False
    pd.DataFrame.mask = mask
    pd.DataFrame.flatten_cols = flatten_cols

    users, ratings = load_data()

    genre_occurences, movielens, movies = binary_feature(users, ratings)
    explore_dataset(users, movies, ratings)

    movies_ratings = get_movie_ratings(movies, ratings)

    # Build the CF model and train it
    model = build_model(users, movies, ratings, embedding_dim=30, init_stddev=0.5)
    model.train(num_iterations=1000, learning_rate=10.)

    # Inspecting the Embeddings
    DOT = 'dot'
    COSINE = 'cosine'

    # Movie Nearest neighbors
    movie_neighbors(movies, model, "Aladdin", DOT)
    movie_neighbors(movies, model, "Aladdin", COSINE)

    movie_embedding_norm(movies, movies_ratings, model)

    # Same as before but with low init_stddev
    model_lowinit = build_model(users, movies, ratings, embedding_dim=30,
                                init_stddev=0.05)  # changed from 0.5 (sd of the random initial embeddings)
    model_lowinit.train(num_iterations=1000, learning_rate=10.)
    movie_neighbors(movies, model_lowinit, "Aladdin", DOT)
    movie_neighbors(movies, model_lowinit, "Aladdin", COSINE)
    movie_embedding_norm(movies, movies_ratings, [model, model_lowinit])  # Comparing previous model with new model

    # Embedding visualization
    genre_filter, genre_chart = filter_and_chart()
    tsne_movie_embeddings(model_lowinit, movies, genre_filter, genre_chart)

    # Build a regularized Matrix Factorization model and train it
    reg_model = build_regularized_model(users, movies, ratings, regularization_coeff=0.1, gravity_coeff=1.0,
                                        embedding_dim=35,
                                        init_stddev=.05)
    reg_model.train(num_iterations=2000, learning_rate=20.,
                    plot_results=False)  # issue here plot_results=True, metric = 3
    user_recommendations(USER_RATINGS, movies, ratings, reg_model, DOT, exclude_rated=True, k=10)

    # Inspect the results of the new model
    movie_neighbors(movies, reg_model, "Aladdin", DOT)
    movie_neighbors(movies, reg_model, "Aladdin", COSINE)
    movie_embedding_norm(movies, movies_ratings, [model, model_lowinit, reg_model])  # Compare all
    tsne_movie_embeddings(reg_model, movies, genre_filter, genre_chart)  # Visualize the embeddings


if __name__ == '__main__':
    main()
