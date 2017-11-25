require 'matrix'

# simple function to do tests
def assert(test,mess="assertion failed")
  raise mess unless test
end

class Matrix
	def hadamard_product(m)
    combine(m){|a, b| a * b}
  end
  alias_method :entrywise_product, :hadamard_product
end


# Matrix element wise matrix multiplication
# Hadamard product (matrices):
# https://en.wikipedia.org/wiki/Hadamard_product_(matrices)
def element_multiplication(m1, m2)
	m3 = Matrix.build(m1.row_count, m1.column_count) {|r, c| m1[r, c] * m2[r, c]}
	return m3
end

# Summation of all values in a matrix
def element_summation(m)
	s = 0
	m.each {|x| s += x}
	return s
end

# a confusion matrix illustrates the accuracy of a classifier:
# values in the diagonal of the classifier are correctly classified.
# https://en.wikipedia.org/wiki/Confusion_matrix
def confusion_matrix(expected, predicted)
	expected = expected.to_a.map {|x| x.index(x.max)}
	predicted = predicted.to_a.map {|x| x.index(x.max)}

	n = (expected + predicted).uniq.length
	cm = Matrix.build(n){0}.to_a
	expected.zip(predicted).map {|x, y| cm[x][y]+=1}

	return Matrix.rows(cm)
end

def accuracy(m)
	sum=0.0
	m.each do  |e|
			sum+=e

	end
	return m.trace.to_f/sum
end




#The actual neural network
class NeuralNetwork

	attr_reader :n_inputs, :n_outputs, :weights, :layer_sizes

	def initialize(n_inputs, n_outputs,inner_layers=[5],alpha=0.1)
		# For the sake of simplicity feel free to hardcode
		# parameters. The goal is a working feedforward neral
		# network. One hidden layer, one input layer, and one
		# output layer are enough to achieve 99% accuracy on
		# the data set.
		@n_layers=inner_layers.size+1
		@layer_sizes=[n_inputs] + inner_layers + [n_outputs]
		@weights=Array.new(@n_layers)
		@deltas=Array.new(@n_layers)
		@x=Array.new(@n_layers+1)
		0.upto(@n_layers-1) do |layer|
			@weights[layer]=Matrix.build(@layer_sizes[layer]+1,@layer_sizes[layer+1]) {rand-0.5}
		end

		@n_inputs=n_inputs
		@n_outputs=n_outputs
		@alpha=alpha
	end

	##############################################
	def train(x, y)
		# the training method that updates the internal weights
		# using the predict
		predict(x)
		back_propagate(y)
		update_weights
	end

	##############################################


	##############################################
	def predict(x)
		@x[0]=x
		propagate
		return @x[@n_layers]
	end

	##############################################
	protected

		def g(x)
			1/(1+Math.exp(-x))
		end

		##############################################
		def propagate
			# applies the input to the network
			# this is the forward propagation step
			0.upto(@n_layers-1) do |layer|
				propagate_layer(layer)
			end
		end

		def propagate_layer(layer)
			x_with_one=Matrix.columns(@x[layer].transpose.to_a << Array.new(@x[layer].row_count,1))
			@x[layer+1]=(x_with_one*@weights[layer]).map { |e| g(e) }
		end

		##############################################
		def back_propagate(y)
			# goes backwards and finds the weights
			# that need to be tuned
			back_propagate_last_layer(y)
			n_layers-2.downto(0) do |layer|
				back_propagate_hidden_layer(layer)
			end
		end

		def back_propagate_hidden_layer(layer)
		end

		def back_propagate_last_layer(y)
			w=@weights[@n_layers-1]
			o=@x[@n_layers]
		end
end


# unit tests
puts "doing neuro unit tests.."

m = NeuralNetwork.new(3,7,[])
assert(m.layer_sizes == [3,7])
assert(m.weights[0].row_count == 4)
assert(m.weights[0].column_count == 7)

m = NeuralNetwork.new(3,7)
assert(m.layer_sizes == [3,5,7])
assert(m.weights[0].row_count == 4)
assert(m.weights[0].column_count == 5)
assert(m.weights[1].row_count == 6)
assert(m.weights[1].column_count == 7)

# accuracy
a=Matrix[[1,2,3],[1,2,3],[1,2,3]]
assert(accuracy(a)==1.0/3)

puts "unit tests complete"

puts "element_multiplication"

a.entrywise_product(a)