require 'e'
require 'xlua'

cmd = torch.CmdLine()
cmd:text('Options:')
cmd:option('--dataset', 0, 'size of dataset to use (0 = all)')
cmd:option('--minWordFreq', 1, 'minimum frequency of words kept in vocab')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--hiddenSize', 300, 'number of hidden units in LSTM')
cmd:option('--learningRate', 0.05, 'learning rate at t=0')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--minLR', 0.00001, 'minimum learning rate')
cmd:option('--saturateEpoch', 20, 'epoch at which linear decayed LR will reach minLR')
cmd:option('--maxEpoch', 50, 'maximum number of epochs to run')
cmd:option('--batchSize', 25, 'number of inputs per batch')
cmd:option('--maxExampleLen', 25, 'max length of sentences used in training')

cmd:text()
options = cmd:parse(arg)

if options.dataset == 0 then
  options.dataset = nil
end

-- Check for LookupTable
assert(options.batchSize <= options.maxExampleLen)

-- Data
print("-- Loading dataset")
dataset = e.DataSet("data/cornell_movie_dialogs_" .. (options.dataset or "full") .. ".t7",
                    e.CornellMovieDialogs("data/cornell_movie_dialogs"),
                    {
                      loadFirst = options.dataset,
                      minWordFreq = options.minWordFreq,
                      maxExampleLen = options.maxExampleLen
                    })

print("\nDataset stats:")
print("  Vocabulary size: " .. dataset.wordsCount)
print("         Examples: " .. #dataset)

-- Model
model = e.Seq2Seq(dataset.wordsCount, options.hiddenSize)
model.goToken = dataset.goToken
model.eosToken = dataset.eosToken

-- Training parameters
model.criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
model.learningRate = options.learningRate
model.momentum = options.momentum
local decayFactor = (options.minLR - options.learningRate) / options.saturateEpoch
local minMeanError = nil

-- Enabled CUDA
if options.cuda then
  require 'cutorch'
  require 'cunn'
  dataset:cuda()
  model:cuda()
end

-- Run the experiment

for epoch = 1, options.maxEpoch do
  print("\n-- Epoch " .. epoch .. " / " .. options.maxEpoch)
  print("")

  local errors = torch.Tensor(#dataset)
  local timer = torch.Timer()

  for i, inputs, targets in dataset:batches(options.batchSize) do
    local err = model:train(inputs, targets)
    errors[i] = err
    xlua.progress(i, #dataset)
  end

  timer:stop()

  print("\nFinished in " .. timer:time().real .. ' seconds. ' .. (#dataset / timer:time().real) .. ' examples/sec.')
  print("\nEpoch stats:")
  print("           LR= " .. model.learningRate)
  print("  Errors: min= " .. errors:min())
  print("          max= " .. errors:max())
  print("       median= " .. errors:median()[1])
  print("         mean= " .. errors:mean())
  print("          std= " .. errors:std())

  if errors:mean() ~= errors:mean() then
    print("\n!! Error is NaN. Bug? Exiting.")
    break
  end

  -- Save the model if we improved.
  if minMeanError == nil or errors:mean() < minMeanError then
    print("\n(Saving model ...)")
    torch.save("data/model.t7", model)
    minMeanError = errors:mean()
  end

  model.learningRate = model.learningRate + decayFactor
  model.learningRate = math.max(options.minLR, model.learningRate)

  collectgarbage()
end

-- Load testing script
require "eval"