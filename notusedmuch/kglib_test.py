
import kglib.kgcn.core.model as model
import kglib.kgcn.learn.classify as classify
import tensorflow as tf
import grakn.client

training_keyspace= "dm_graph"
neighbour_sample_sizes= [7,2,2]
features_size= 10
example_things_features_size= 5
aggregated_size= 20
embedding_size= 32
batch_size= 10
learning_rate= .1
num_classes=

URI = "172.16.253.242:48555"

client = grakn.client.GraknClient(uri=URI)
session = client.session(keyspace=training_keyspace)
transaction = session.transaction().write()

kgcn = model.KGCN(neighbour_sample_sizes,
                  features_size,
                  example_things_features_size,
                  aggregated_size,
                  embedding_size,
                  transaction,
                  batch_size)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

classifier = classify.SupervisedKGCNMultiClassSingleLabelClassifier(kgcn,
                                               optimizer, 
                                               num_classes, 
                                               log_dir,
                                               max_training_steps=max_training_steps)

training_feed_dict = classifier.get_feed_dict(session, 
                                              training_things, 
                                              labels=training_labels)

classifier.train(training_feed_dict)

transaction.close()
session.close()
