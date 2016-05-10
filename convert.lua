require 'nn'
require 'cunn'
require 'cudnn'
require './Models/BinaryLinear.lua'
require './Models/BinarizedNeurons'
require './Models/BatchNormalizationShiftPow2'
require 'bit'

function pack(x)
	N = x:nElement()
	x = x:view(N)
	x = x:add(1):div(2):round():int()

	size = math.floor((N-1) / 32) + 1
	tmp = torch.IntTensor(size)
	fill = false
	for i = 0, size-1 do
		acc = 0
		local mask = 1
		for j = 1, 32 do
			k = i*32 + j
			if (k > N) then
				fill = true
			end
			if fill or (x[k] == 1) then
				acc = bit.bor(acc, mask)
			end
			mask = bit.lshift(mask, 1)
		end
		tmp[i+1] = acc
	end

	return tmp
end

model = torch.load('Net')
-- transfer from GPU to CPU if needed
model = model:float()
-- this is important
model:evaluate()

f = torch.DiskFile('weights.bin', 'w')
f:binary()

-- the layers that contain weights
idx = {2, 6, 10, 14}
-- batch normalization layers
batchNormLayers = {3, 7, 11}

for i, v in ipairs(idx) do
	m = model.modules[v]
	w = m:binarized(false)

	if batchNormLayers[i] then
		mean = model.modules[batchNormLayers[i]].running_mean
		mean = torch.round(mean:div(2)):int()
	else
		mean = nil
	end

	-- write the weights and biases
	-- set of weights for an output bit must be 32-bit aligned
	for i = 1, w:size(1) do
		f:writeInt(pack(w[{ {i}, {} }]):storage())
		if mean then
			f:writeInt(mean[i])
		end
	end
end

f:close()
