require 'matrix'
#require 'nmatrix'

# simple function to do tests
def assert(test,mess="assertion failed")
  raise mess unless test
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

# adds the same row to every row in matrix
# +row+ is a single row matrix
def add_row_to_matrix(m,row)
	#assert(m.column_count==row.column_count,"matrix has #{m.column_count} and row has #{row.column_count} columns")
	#assert(row.row_count==1)
	return Matrix.build(m.row_count,m.column_count) do |r,c|
			# puts "#{r},#{c}: \"#{m[r,c]}\",\"#{row[0,c]}\""
			# puts m[r,c]+row[0,c]
			 m[r,c]+row[0,c]
		end
end

# sums matrix into a single row matrix
def total_row(m)
	# sums all rows
	Matrix.row_vector(m.row_vectors.each.reduce(:+))

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

# the proportion of correctly classified values in the confusion matrix
def accuracy(m)
	sum=0.0
	m.each do  |e|
			sum+=e

	end
	return m.trace.to_f/sum
end


# simple Gaussian random generator
def gaussian(mean, stddev)
  theta = 2 * Math::PI * rand
  rho = Math.sqrt(-2 * Math.log(rand))
  scale = stddev * rho
  x = mean + scale * Math.cos(theta)

  return x
end



# The actual neural network
class NeuralNetwork

	attr_reader :n_inputs, :n_outputs, :weights, :layer_sizes

	def initialize(n_inputs, n_outputs,inner_layers=[5],alpha=0.02,lmbda=0.01)
		# alpha is used to calculate the speed of lerning
		# lmbda is used for regularization (with L2-norm)
		@n_layers=inner_layers.size+2
		@layer_sizes=[n_inputs] + inner_layers + [n_outputs]
		@weights=Array.new(@n_layers)
		@biases=Array.new(@n_layers)
		@deltas=Array.new(@n_layers)
		@x=Array.new(@n_layers)
		1.upto(@n_layers-1) do |layer|
			@weights[layer]=Matrix.build(@layer_sizes[layer-1],@layer_sizes[layer]) {gaussian(0, 1/Math.sqrt(@layer_sizes[layer-1]))}
			@biases[layer]= Matrix.build(1,@layer_sizes[layer]) {gaussian(0, 1)}
		end

		@n_inputs=n_inputs
		@n_outputs=n_outputs
		@alpha=alpha
		@lmbda=lmbda
		#checkBiases
		#checkWeights
	end

	##############################################
	def train(x, y)
		# the training method that updates the internal weights
		# for a batch (single training step)
		predict(x)

		back_propagate_deltas(y)

		@eta=@alpha*Math.sqrt(x.column_count) # rough adjustment of learning speed
		update_weights_and_biases
		calculate_cost(y)
	end


	##############################################
	def predict(x)
		@x[0]=x
		propagate
		return @x[@n_layers-1]

	end

	# cost function that can be used to show progress even if accuracy stays unchanges
	def calculate_cost(y)
		# we use here L2-norm even though backpropagation may be based on another cost function
		l2=element_summation((@x[@n_layers-1]-y).map {|e| e*e})
		return l2
	end

	##############################################
	protected

		# debug methods
				def checkRep
					checkWeights
					checkBiases
					checkDeltas
					checkXes
				end

				def checkWeights
					1.upto(@n_layers-1) do |layer|
						checkW(layer)
					end
				end

				def checkW(layer)
					w=@weights[layer]
					assert(w.row_count==@layer_sizes[layer-1])
					assert(w.column_count==@layer_sizes[layer])
				end

				def checkBiases
					1.upto(@n_layers-1) do |layer|
						checkB(layer)
					end
				end

				def checkB(layer)
					b=@biases[layer]
					assert(b.column_count==@layer_sizes[layer], "B in layer #{layer} has #{b.row_count} rows instead of #{@layer_sizes[layer]}")
					assert(b.row_count==1)
					b.each do |e| assert(e!=nil) end
				end

				def checkDeltas
					1.upto(@n_layers-1) do |layer|
						checkD(layer)
					end
				end

				def checkD(layer)
					d=@deltas[layer]
					assert(d.row_count==@x[0].row_count)
					assert(d.column_count==@layer_sizes[layer])
				end

				def checkXes
					0.upto(@n_layers-1) do |layer|
						checkX(layer)
					end
				end

				def checkX(layer)
					x=@x[layer]
					assert(x.row_count==@x[0].row_count,"X in layer #{layer} has #{x.row_count} rows instead of #{@x[0].row_count}")
					assert(x.column_count==@layer_sizes[layer],"X in layer #{layer} has #{x.column_count} rows instead of #{@layer_sizes[layer]}")
				end


		# z is scalar
		def g(z)
			sigmoid(z)
		end

		def sigmoid(z)
			1/(1+Math.exp(-z))
		end

		# derivative of simoid(z) (with respect to z) express in terms of simoid(z)
		# g'(z)=g(z)*(1-g(z))
		def sig_prime(a_vec)
			a_vec.map { |a| a*(1-a) }
		end

		##############################################
		def propagate
			# applies the input to the network
			# this is the forward propagation step

			#checkBiases
			#checkWeights
			1.upto(@n_layers-1) do |layer|
				propagate_layer(layer)
			end
			#checkXes
		end

		def propagate_layer(layer)

			@x[layer]=add_row_to_matrix((@x[layer-1]*weights[layer]),@biases[layer]).map { |e| g(e) }
		end

		##############################################
		def back_propagate_deltas(y)
			# goes backwards and finds the deltas
			# to tune weights

			#checkBiases
			#checkWeights
			#checkXes
			back_propagate_last_layer(y)
			(@n_layers-2).downto(1) do |layer|
				back_propagate_hidden_layer(layer)
			end
			#checkDeltas
		end

		def back_propagate_hidden_layer(layer)
			#calculates deltas
			#for sigmoid actuation

			#checkD(layer+1)
			@deltas[layer]=element_multiplication( @deltas[layer+1]*@weights[layer+1].transpose , sig_prime(@x[layer]) )
			#checkD(layer)
		end

		def back_propagate_last_layer(y)
			#using cross-entropy cost function with sigmoid actuation
			@deltas[@n_layers-1]=@x[@n_layers-1]-y
			#checkD(@n_layers-1)
		end

		def update_weights_and_biases()

			1.upto(@n_layers-1) do |layer|
				@biases[layer]-=total_row(@deltas[layer]).map {|e| e*@eta}
				@weights[layer]=(@weights[layer].map {|e| e*(1-@eta*@lmbda)})-((@x[layer-1].transpose*@deltas[layer]).map {|e| e*@eta})
			end
		end
end


# unit tests
puts "doing neuro unit tests.."

m = NeuralNetwork.new(3,7,[])
assert(m.layer_sizes == [3,7])
assert(m.weights[1].row_count == 3)
assert(m.weights[1].column_count == 7)

m = NeuralNetwork.new(3,7)
assert(m.layer_sizes == [3,5,7])
assert(m.weights[1].row_count == 3)
assert(m.weights[1].column_count == 5)
assert(m.weights[2].row_count == 5)
assert(m.weights[2].column_count == 7)



# accuracy
a=Matrix[[1,2,3],[1,2,3],[1,2,3]]
assert(accuracy(a)==1.0/3)

puts "unit tests complete"

