import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM

def make_parallel(inner_model, gpu_count):
	"""Creates a new wrapper model that consists of multiple replicas of
	the original model placed on different GPUs.
	"""
	# Slice inputs. Slice inputs on the CPU to avoid sending a copy
	# of the full inputs to all GPUs. Saves on bandwidth and memory.
	input_slices = {name: tf.split(x, gpu_count)
					for name, x in zip(inner_model.input_names,
									   inner_model.inputs)}

	output_names = inner_model.output_names
	outputs_all = []
	for i in range(len(inner_model.outputs)):
		outputs_all.append([])

	# Run the model call() on each GPU to place the ops there
	for i in range(gpu_count):
		with tf.device('/gpu:%d' % i):
			with tf.name_scope('tower_%d' % i):
				# Run a slice of inputs through this replica
				zipped_inputs = zip(inner_model.input_names,
									inner_model.inputs)
				inputs = [
					KL.Lambda(lambda s: input_slices[name][i],
							  output_shape=lambda s: (None,) + s[1:])(tensor)
					for name, tensor in zipped_inputs]
				# Create the model replica and get the outputs
				outputs = inner_model(inputs)
				if not isinstance(outputs, list):
					outputs = [outputs]
				# Save the outputs for merging back together later
				for l, o in enumerate(outputs):
					outputs_all[l].append(o)

	# Merge outputs on CPU
	with tf.device('/cpu:0'):
		merged = []
		for outputs, name in zip(outputs_all, output_names):
			# If outputs are numbers without dimensions, add a batch dim.
			def add_dim(tensor):
				"""Add a dimension to tensors that don't have any."""
				if K.int_shape(tensor) == ():
					return KL.Lambda(lambda t: K.reshape(t, [1, 1]))(tensor)
				return tensor
			outputs = list(map(add_dim, outputs))

			# Concatenate
			merged.append(KL.Concatenate(axis=0, name=name)(outputs))
# 	return merged
	return KM.Model(inputs=inner_model.inputs, outputs=merged)