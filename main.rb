#!/usr/bin/env ruby

# some requirements
require 'set'
require 'ruby-prof'
load 'neuro.rb'

# We will use the iris data set based which contains observations of
# flower petal and sepal size to predict the flower genus
# Dataset: http://en.wikipedia.org/wiki/Iris_flower_data_set

data = File.readlines("iris.csv").map {|l| l.chomp.split(',') }

class_label = {
  "s" => [1, 0, 0],
  "c" => [0, 1, 0],
  "v" => [0, 0 ,1]
}

x = data.map {|x| x[0,4].map(&:to_f) }
y = data.map {|x| class_label[x[4]] }

# Normalize data values before feeding into network
normalize = -> (val, high, low) {  (val - low) / (high - low) } # maps input to float between 0 and 1

features = (0..3).map do |i|
    x.map {|row| row[i] }
end

x.map! do |row|
    row.map.with_index do |val, j|
        max, min = features[j].max, features[j].min
        normalize.(val, max, min)
    end
end

# We split the iris data so that a each class has the same
# probability to appear in each sub set (train and test) as
# it has in the full set #stratified sampling.For the iris
# data set the propabilities are equal.

# number of elements per class in the training set.
n_train = 25
n_tries=1
n_epoches=2000
accs=[]

n_tries.times do

    d1 = (0..49).to_a.sample(n_train)
    d2 = (50..99).to_a.sample(n_train)
    d3 = (100..149).to_a.sample(n_train)

    # samples we want in the train set.
    train_samples = (d1 + d2 + d3).to_set

    # samples we want in the test set are
    # those not in the training set.
    test_samples = (0..149).to_set - train_samples

    x_train = Matrix.rows(x.values_at(*train_samples))
    y_train = Matrix.rows(y.values_at(*train_samples))

    x_test = Matrix.rows(x.values_at(*test_samples))
    y_test = Matrix.rows(y.values_at(*test_samples))

    # creates a new neural network
    nn = NeuralNetwork.new(x_test.column_count,y_test.column_count,[5,5],0.002, 0.002)

    # trains the network with the training set
    # and predicts the classes of the unseen
    # test data

  starttime=Time.now
  #RubyProf.start

      n_epoches.times do
        nn.train(x_train, y_train)
      end

  #result = RubyProf.stop
  puts Time.now-starttime


    # print a flat profile to text
    # printer = RubyProf::GraphPrinter.new(result)
    # printer.print(STDOUT, {})

  y_hat = nn.predict(x_test)
  cm = confusion_matrix(y_test, y_hat)
  acc = accuracy(cm)
  accs<<=acc
  # prints the networks "accuracy" in predicting
  # the unseen test set.
  p cm.to_s
  p "Accuracy: " + acc.to_s
end

mean_acc=accs.reduce(:+)/n_tries
sd_acc=(accs.inject(0) { |a| (a-mean_acc)**2})/(n_tries-1)

puts "   --- results   ---"
puts "Mean: #{mean_acc}, SD: #{Math.sqrt(sd_acc)}"

File.open("results", 'w') do |file|
    file.write(" \n")
    file.write("   --- results   ---\n")
    file.write("Mean: #{mean_acc}, SD: #{Math.sqrt(sd_acc)}\n")
end