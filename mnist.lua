function pack(x)
	N = x:nElement()
	x = x:view(N)
	x = x:add(1):div(2):round():int()

	size = math.floor((N-1) / 32) + 1
	tmp = torch.IntTensor(size)
	for i = 0, size-1 do
		acc = 0
		local mask = 1
		for j = 1, 32 do
			k = i*32 + j
			if (k > N) then
				break
			end
			if x[k] == 1 then
				acc = bit.bor(acc, mask)
			end
			mask = bit.lshift(mask, 1)
		end
		tmp[i+1] = acc
	end

	return tmp
end

opt = {}
opt.dataset = 'MNIST'

data = require 'Data'
examples = data.TrainData.Data
N = examples:size(1)
height = examples:size(2)
width = examples:size(3)

x = data.TrainData.Data:sign()

f = torch.DiskFile('mnist.bin', 'w')
f:binary()
f:writeInt(N)
f:writeInt(height)
f:writeInt(width)
-- each image must be 32-bit aligned
for i = 1, N do
	f:writeInt(pack(x[{ {i}, {}, {}, {} }]):storage())
end
f:close()

f = torch.DiskFile('labels.bin', 'w')
f:binary()
f:writeInt(data.TrainData.Labels:int():storage())
f:close()
