require 'neuralconvo'
local tokenizer = require "tokenizer"
local list = require "pl.List"

if dataset == nil then
  cmd = torch.CmdLine()
  cmd:text('Options:')
  cmd:option('--cuda', false, 'use CUDA. Training must be done on CUDA')
  cmd:text()
  options = cmd:parse(arg)

  -- Data
  dataset = neuralconvo.DataSet()

  -- Enabled CUDA
  if options.cuda then
    require 'cutorch'
    require 'cunn'
  end
end

if model == nil then
  print("-- Loading model")
  model = torch.load("data/model.t7")
end

-- Word ID tensor to words
function t2w(t)
  local words = {}

  for i = 1, t:size(1) do
    table.insert(words, dataset.id2word[t[i]])
  end

  return words
end

function say(text)
  local wordIds = {}

--  for t, word in tokenizer.tokenize(text) do

  for word in string.gmatch(text, "[^%s]+") do
    local id = dataset.word2id[word:lower()] or dataset.unknownToken
    --print(word, id)
    table.insert(wordIds, id)
  end

  local input = torch.Tensor(list.reverse(wordIds))
  --print("input: ", input)
  local output = model:eval(input)
  --print(output)

  print(">> " .. tokenizer.join(t2w(torch.Tensor(output))))
end
