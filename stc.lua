--[[
Format movie dialog data as a table of line 1:

  { {word_ids of character1}, {word_ids of character2} }

Then flips it around and get the dialog from the other character's perspective:

  { {word_ids of character2}, {word_ids of character1} }

Also builds the vocabulary.
]]-- 

local Stc = torch.class("neuralconvo.Stc")
local xlua = require "xlua"
local tokenizer = require "tokenizer"
local list = require "pl.List"

function Stc:__init(dir)
  options = options or {}

  self.examplesFilename = "data/examples.t7"

  -- Discard words with lower frequency then this
  self.minWordFreq = options.minWordFreq or 1

  -- Maximum number of words in an example sentence
  self.maxExampleLen = options.maxExampleLen or 25

  -- Load only first fews examples (approximately)
  self.loadFirst = options.loadFirst

  self.examples = {}
  self.word2id = {}
  self.id2word = {}
  self.wordFreq = {}
  self.wordsCount = 0
  self.examplesCount = 0
  self.dir = dir

  self.loadFromFile = true
  self:load()

end


function Stc:load()
  local filename = "data/vocab.t7"

  --if path.exists(filename) then
  if nil then
    print("Loading vocabulary from " .. filename .. " ...")
    local data = torch.load(filename)
    self.word2id = data.word2id
    self.id2word = data.id2word
    self.wordsCount = data.wordsCount
    self.goToken = data.goToken
    self.eosToken = data.eosToken
    self.unknownToken = data.unknownToken
    self.examplesCount = data.examplesCount
  else
    print("" .. filename .. " not found")
    if self.loadFromFile then
        print("load from file")
        self:visitFile()    
        self:setFileWordId()
    else
        local conversations = self:visitStc()
        self:visit(conversations)
    end
    print("Writing " .. filename .. " ...")
    torch.save(filename, {
      word2id = self.word2id,
      id2word = self.id2word,
      wordsCount = self.wordsCount,
      goToken = self.goToken,
      eosToken = self.eosToken,
      unknownToken = self.unknownToken,
      examplesCount = self.examplesCount
    })
  end
end

function Stc:visitFile()
   self.goToken = "#START#" 
   self.eosToken = "#END#"
   self.unknownToken = "#UNK#"  

   local conversations = self:visitStc()
   for i, conversation in ipairs(conversations) do
        local input, target = unpack(conversation)
        assert(input ~= "");
        assert(target ~= "");

        local inputIds = {}
        --print(input)
        for word in string.gmatch(input, "[^%s]+") do
            table.insert(inputIds, word)
        end
        --print("input: ", inputIds)
        --for t, word in tokenizer:tokenize(input) do
        --    table.insert(inputIds, word)
        --end
        local targetIds = {}
        --print(target)
        for word in string.gmatch(target, "[^%s]+") do
            table.insert(targetIds, word)
        end
        --for t, word in tokenizer:tokenize(target) do
        --    table.insert(targetIds, word)
        --end

        --print("target: ", targetIds)
        table.insert(self.examples, { torch.IntTensor(inputIds), torch.IntTensor(targetIds) })
   end

   self.examplesCount = #self.examples
   print("examples count: ", self.examplesCount)
   self:writeExamplesToFile()
   self.examples = nil

end

function Stc:setFileWordId()
    vocabName = self.dir .. "vocab"
    print("vocabName: ", vocabName)
    local pvocab = assert(io.open(vocabName, 'r'))

    --self.id2word[1] = "#START#"
    --self.word2id["#START#"] = 1
    --self.id2word[2] = "#END#"
    --self.word2id["#END#"] = 2
    --self.id2word[3] = "#UNK#"
    --self.word2id["#UNK#"] = 3

    self.wordsCount = 0
    local index = 1
    while(1) do
        local v = pvocab:read("*line")
        if (v == nil) then
            break
        end
        print("line: ", v)
        self.wordsCount = self.wordsCount + 1
        index = self.wordsCount
        self.id2word[index] = v
        self.word2id[v] = index
        self.wordFreq[v] = 5
        print(index, v)
    end
end

function Stc:visitStc()
    --local postName = self.dir .. "mini_post_filtered"
    --local commentName = self.dir .. "mini_comment_filtered"
    local postName = self.dir .. "mini_post_index"
    local commentName = self.dir .. "mini_comment_index"
    print(postName)
    local ppost = assert(io.open(postName, 'r'))
    local pcomment = assert(io.open(commentName, 'r'))
    local conversations = {}
    while(1) do
        local conversation = {}
        local post = ppost:read("*line")
        local comment = pcomment:read("*line")
        if (post == nil or comment == nil) then
            break;
        end
        table.insert(conversation, post)
        table.insert(conversation, comment)
        --print(conversation)
        table.insert(conversations, conversation)
    end
    --self:visit(conversations)
    return conversations
end

function Stc:makeWordIds(word)
  word = word:lower()

  local id = self.word2id[word]

  if id then
    self.wordFreq[word] = self.wordFreq[word] + 1
  else
    self.wordsCount = self.wordsCount + 1
    id = self.wordsCount
    self.id2word[id] = word
    self.word2id[word] = id
    self.wordFreq[word] = 1
  end

  return id
end


function Stc:visit(conversations)
  -- Table for keeping track of word frequency
  self.wordFreq = {}
  self.examples = {}

  print("in visit")
  -- Add magic tokens
  self.goToken = self:makeWordIds("<go>") -- Start of sequence
  self.eosToken = self:makeWordIds("<eos>") -- End of sequence
  self.unknownToken = self:makeWordIds("<unknown>") -- Word dropped from vocabulary

  print("-- Pre-processing data")
  total = #conversations

  for i, conversation in ipairs(conversations) do
    print(conversation)
    self:visitConversation(conversation)
    xlua.progress(i, total)
  end

  print("-- Removing low frequency words")

  for i, datum in ipairs(self.examples) do
    self:removeLowFreqWords(datum[1])
    self:removeLowFreqWords(datum[2])
    xlua.progress(i, #self.examples)
  end

  self.wordFreq = nil

  self.examplesCount = #self.examples
  self:writeExamplesToFile()
  self.examples = nil

  collectgarbage()
end

function Stc:writeExamplesToFile()
  print("Writing " .. self.examplesFilename .. " ...")
  local file = torch.DiskFile(self.examplesFilename, "w")

  for i, example in ipairs(self.examples) do
    print(i)
    print(example)
    file:writeObject(example)
    xlua.progress(i, #self.examples)
  end

  file:close()
end

function Stc:batches(size)
  local file = torch.DiskFile(self.examplesFilename, "r")
  file:quiet()
  local done = false

  return function()
    if done then
      return
    end

    local examples = {}

    for i = 1, size do
      local example = file:readObject()
      if example == nil then
        done = true
        file:close()
        return examples
      end
      table.insert(examples, example)
    end

    return examples
  end
end

function Stc:removeLowFreqWords(input)
  for i = 1, input:size(1) do
    local id = input[i]
    local word = self.id2word[id]

    if word == nil then
      -- Already removed
      input[i] = self.unknownToken

    elseif self.wordFreq[word] < self.minWordFreq then
      input[i] = self.unknownToken
      
      self.word2id[word] = nil
      self.id2word[id] = nil
      self.wordsCount = self.wordsCount - 1
    end
  end
end

function Stc:visitConversation(conversation)

    local input, target = unpack(conversation)
    print(input)
    print(target)

    if target then
      local inputIds = self:visitText(input)
      local targetIds = self:visitText(target)

      if inputIds and targetIds then
        -- Revert inputs
        inputIds = list.reverse(inputIds)

        table.insert(targetIds, 1, self.goToken)
        table.insert(targetIds, self.eosToken)

        table.insert(self.examples, { torch.IntTensor(inputIds), torch.IntTensor(targetIds) })
      end
    end
end

function Stc:visitText(text, additionalTokens)
  local words = {}
  additionalTokens = additionalTokens or 0

  if text == "" then
    return
  end

  print(text)
  for t, word in tokenizer.tokenize(text) do
    id = self:makeWordIds(word)
    print(id, word)
    table.insert(words, id)
    --table.insert(words, self:makeWordIds(word))
    -- Only keep the first sentence
    if t == "endpunct" or #words >= self.maxExampleLen - additionalTokens then
      break
    end
  end

  if #words == 0 then
    return
  end

  return words
end

