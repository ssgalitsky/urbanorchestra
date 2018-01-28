import os
import tensorflow as tf
import audioset.vggish_input as vggish_input
import audioset.vggish_params as vggish_params
import audioset.vggish_postprocess as vggish_postprocess
import audioset.vggish_slim as vggish_slim


AUDIOSET_DIR = os.path.join(os.path.dirname(__file__), 'audioset')
PCA_PARAMS_PATH = os.path.join(AUDIOSET_DIR, 'vggish_pca_params.npz')
MODEL_PATH = os.path.join(AUDIOSET_DIR, 'vggish_model.ckpt')


def extract_vggish_embedding(audio_data, fs):
    examples_batch = vggish_input.waveform_to_examples(audio_data, fs)

    # Prepare a postprocessor to munge the model embeddings.
    pproc = vggish_postprocess.Postprocessor(PCA_PARAMS_PATH)

    # If needed, prepare a record writer to store the postprocessed embeddings.
    #writer = tf.python_io.TFRecordWriter(
    #    FLAGS.tfrecord_file) if FLAGS.tfrecord_file else None

    with tf.Graph().as_default(), tf.Session() as sess:
      # Define the model in inference mode, load the checkpoint, and
      # locate input and output tensors.
      vggish_slim.define_vggish_slim(training=False)
      vggish_slim.load_vggish_slim_checkpoint(sess, MODEL_PATH)
      features_tensor = sess.graph.get_tensor_by_name(
          vggish_params.INPUT_TENSOR_NAME)
      embedding_tensor = sess.graph.get_tensor_by_name(
          vggish_params.OUTPUT_TENSOR_NAME)

      # Run inference and postprocessing.
      [embedding_batch] = sess.run([embedding_tensor],
                                   feed_dict={features_tensor: examples_batch})
      postprocessed_batch = pproc.postprocess(embedding_batch)

      # Write the postprocessed embeddings as a SequenceExample, in a similar
      # format as the features released in AudioSet. Each row of the batch of
      # embeddings corresponds to roughly a second of audio (96 10ms frames), and
      # the rows are written as a sequence of bytes-valued features, where each
      # feature value contains the 128 bytes of the whitened quantized embedding.

    return postprocessed_batch
