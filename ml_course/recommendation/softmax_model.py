from __future__ import print_function
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import ml_course.recommendation.collaborative_filtering_clean as cf

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.float_format = '{:.3f}'.format
pd.options.display.max_rows = 15
pd.options.display.width = 0
pd.DataFrame.mask = cf.mask
pd.DataFrame.flatten_cols = cf.flatten_cols

users, ratings = cf.load_data()
genre_occurences, movielens, movies = cf.binary_feature(users, ratings)
movies_ratings = cf.get_movie_ratings(movies, ratings)
USER_RATINGS = False
DOT = 'dot'
COSINE = 'cosine'
genre_filter, genre_chart = cf.filter_and_chart()

# SoftMax model
"""
Softmax model that predicts whether a given user has rated a movie. The model will take as input a feature vector x 
representing the list of movies the user has rated. Start from the ratings DataFrame, which is grouped by by user_id.

Create a function that generates an example batch, such that each example contains the following features:
* movie_id: A tensor of strings of the movie ids that the user rated.
* genre: A tensor of strings of the genres of those movies
* year: A tensor of strings of the release year.
"""

rated_movies = (ratings[["user_id", "movie_id"]].groupby("user_id", as_index=False).aggregate(lambda x: list(x)))
rated_movies.head()

# Batch generation code
years_dict = {movie: year for movie, year in zip(movies["movie_id"], movies["year"])}
genres_dict = {movie: genres.split('-') for movie, genres in zip(movies["movie_id"], movies["all_genres"])}


def make_batch(ratings, batch_size):
    """Creates a batch of examples.
    Args:
      ratings: A DataFrame of ratings such that examples["movie_id"] is a list of movies rated by a user.
      batch_size: The batch size.
    """

    def pad(x, fill):
        return pd.DataFrame.from_dict(x).fillna(fill).values

    movie = []
    year = []
    genre = []
    label = []
    for movie_ids in ratings["movie_id"].values:
        movie.append(movie_ids)
        genre.append([x for movie_id in movie_ids for x in genres_dict[movie_id]])
        year.append([years_dict[movie_id] for movie_id in movie_ids])
        label.append([int(movie_id) for movie_id in movie_ids])
    features = {
        "movie_id": pad(movie, ""),
        "year": pad(year, ""),
        "genre": pad(genre, ""),
        "label": pad(label, -1)
    }
    batch = (
        tf.data.Dataset.from_tensor_slices(features).shuffle(1000).repeat().batch(
            batch_size).make_one_shot_iterator().get_next())
    return batch


def select_random(x):
    """Selectes a random elements from each row of x."""

    def to_float(x):
        return tf.cast(x, tf.float32)

    def to_int(x):
        return tf.cast(x, tf.int64)

    batch_size = tf.shape(x)[0]
    rn = tf.range(batch_size)
    nnz = to_float(tf.count_nonzero(x >= 0, axis=1))
    rnd = tf.random_uniform([batch_size])
    ids = tf.stack([to_int(rn), to_int(nnz * rnd)], axis=1)
    return to_int(tf.gather_nd(x, ids))


# Loss function
def softmax_loss(user_embeddings, movie_embeddings, labels):
    """Returns the cross-entropy loss of the softmax model.
    Args:
      user_embeddings: A tensor of shape [batch_size, embedding_dim].
      movie_embeddings: A tensor of shape [num_movies, embedding_dim].
      labels: A tensor of [batch_size], such that labels[i] is the target label
        for example i.
    Returns:
      The mean cross-entropy loss.
    """
    # Verify that the embddings have compatible dimensions
    user_emb_dim = user_embeddings.shape[1].value
    movie_emb_dim = movie_embeddings.shape[1].value
    if user_emb_dim != movie_emb_dim:
        raise ValueError(
            "The user embedding dimension %d should match the movie embedding "
            "dimension % d" % (user_emb_dim, movie_emb_dim))

    logits = tf.matmul(user_embeddings, movie_embeddings, transpose_b=True)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return loss


# Build Softmax model
def build_softmax_model(rated_movies, embedding_cols, hidden_dims):
    """Builds a Softmax model for MovieLens.
    Args:
      rated_movies: DataFrame of traing examples.
      embedding_cols: A dictionary mapping feature names (string) to embedding column objects. This will be used in
      tf.feature_column.input_layer() to create the input layer.
      hidden_dims: int list of the dimensions of the hidden layers.
    Returns:
      A CFModel object.
    """

    def create_network(features):
        """Maps input features dictionary to user embeddings.
        Args:
          features: A dictionary of input string tensors.
        Returns:
          outputs: A tensor of shape [batch_size, embedding_dim].
        """
        # Create a bag-of-words embedding for each sparse feature.
        inputs = tf.feature_column.input_layer(features, embedding_cols)
        # Hidden layers.
        input_dim = inputs.shape[1].value
        for i, output_dim in enumerate(hidden_dims):
            w = tf.get_variable(
                "hidden%d_w_" % i, shape=[input_dim, output_dim],
                initializer=tf.truncated_normal_initializer(
                    stddev=1. / np.sqrt(output_dim))) / 10.
            outputs = tf.matmul(inputs, w)
            input_dim = output_dim
            inputs = outputs
        return outputs

    train_rated_movies, test_rated_movies = cf.split_dataframe(rated_movies)
    train_batch = make_batch(train_rated_movies, 200)
    test_batch = make_batch(test_rated_movies, 100)

    with tf.variable_scope("model", reuse=False):
        # Train
        train_user_embeddings = create_network(train_batch)
        train_labels = select_random(train_batch["label"])
    with tf.variable_scope("model", reuse=True):
        # Test
        test_user_embeddings = create_network(test_batch)
        test_labels = select_random(test_batch["label"])
        movie_embeddings = tf.get_variable(
            "input_layer/movie_id_embedding/embedding_weights")

    test_loss = softmax_loss(test_user_embeddings, movie_embeddings, test_labels)
    train_loss = softmax_loss(train_user_embeddings, movie_embeddings, train_labels)
    _, test_precision_at_10 = tf.metrics.precision_at_k(
        labels=test_labels,
        predictions=tf.matmul(test_user_embeddings, movie_embeddings, transpose_b=True),
        k=10)

    metrics = (
        {"train_loss": train_loss, "test_loss": test_loss},
        {"test_precision_at_10": test_precision_at_10}
    )
    embeddings = {"movie_id": movie_embeddings}
    return cf.CFModel(embeddings, train_loss, metrics)


# Train the Softmax model
"""
Train the softmax model and set the following hyperparameters:
* learning rate
* number of iterations. Note: you can run softmax_model.train() again to continue training the model from its current state.
* input embedding dimensions (the input_dims argument)
* number of hidden layers and size of each layer (the hidden_dims argument)
"""


# Create feature embedding columns
def make_embedding_col(key, embedding_dim):
    categorical_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=key, vocabulary_list=list(set(movies[key].values)), num_oov_buckets=0)
    return tf.feature_column.embedding_column(
        categorical_column=categorical_col, dimension=embedding_dim, combiner='mean')
    # default initializer: trancated normal with stddev=1/sqrt(dimension)


with tf.Graph().as_default():
    softmax_model = build_softmax_model(
        rated_movies,
        embedding_cols=[
            make_embedding_col("movie_id", 35),
            make_embedding_col("genre", 3),
            make_embedding_col("year", 2),
        ],
        hidden_dims=[35])

softmax_model.train(learning_rate=8., num_iterations=3000, optimizer=tf.train.AdagradOptimizer, plot_results=False)

reg_model = cf.build_regularized_model(users, movies, ratings, regularization_coeff=0.1, gravity_coeff=1.0,
                                       embedding_dim=35,
                                       init_stddev=.05)

reg_model.train(num_iterations=2000, learning_rate=20., plot_results=False)  # issue here plot_results=True, metric = 3
cf.user_recommendations(USER_RATINGS, movies, ratings, reg_model, DOT, exclude_rated=True, k=10)

# Inspect the embeddings
cf.movie_neighbors(movies, softmax_model, "Aladdin", DOT)
cf.movie_neighbors(movies, softmax_model, "Aladdin", COSINE)
cf.movie_embedding_norm(movies, movies_ratings, [reg_model, softmax_model])
cf.tsne_movie_embeddings(softmax_model, movies, genre_filter, genre_chart)
