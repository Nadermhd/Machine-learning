import torch 
import torch.nn as nn
import torch.nn.functional as F 

class conv_module(nn.Module):
	def __init__(self, inch, outch):
		super(conv_module, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(inch, outch, 3, passing=1),
			nn.BachNorm2d(outch),
			nn.Relu(inplace=True),

			nn.Conv2d(inch, outch, 3, passing=1),
			nn.BachNorm2d(outch),
			nn.Relu(inplace=True)
			)

	def forward(self, x):
		x = self.conv(x)
		return x

class inconv(nn.Module):
	"""docstring for inconv"""
	def __init__(self, inch, outch):
		super(inconv, self).__init__()
		self.conv = conv_module(inch, outch)

	def forward(self, x):
		x = self.conv(x)
		return x

class down(nn.Module):
	"""docstring for down"""
	def __init__(self, inch, outch):
		super(down, self).__init__()
		self.mpconv = nn.Sequential(
			nn.MaxPool2d(2),
			conv_module(inch, outch)
			)
		
	def forward(self, x):
		x = self.mpconv(x)
		return x

class up(nn.Module):
	"""docstring for up"""
	def __init__(self, inch, outch, bilinear=True):
		super(up, self).__init__()
		if bilinear=True:
			self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corner=True)
		else:
			self.up = nn.ConvTranspose2d(inch//2, outch//2, 2, stride=2)
		self.conv = conv_module(inch, outch)

	def forward(self, x1, x2):
		x1 = self.up(x1)

		x = torch.cat([x1, x2], dim=1)
		x = self.conv(x)
		return x

class out(nn.Module):
	"""docstring for out"""
	def __init__(self, inch, outch):
		super(out, self).__init__()
		self.conv = nn.Conv2d(inch, outch, 1)

	def forward(self, x):
		x = self.conv(x)
		return x

class Net(nn.Module):
	"""docstring for Net"""
	def __init__(self, in_ch, n_cls):
		super(Net, self).__init__()
		
		self.inc = inconv(in_ch, 64)
		self.down1 = down(64, 128)
		self.down2 = down(128, 256)
		self.down3 = down(256, 512)
		self.down4 = down(512, 1024)

		self.up1 = up(1024, 512)
		self.up2 = up(512, 256)
		self.up3 = up(256, 128)
		self.up4 = up(128, 64)

		self.outc = out(64, n_cls)


	def forward(self, x):
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)

		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		x = self.out(x)

		return F.sigmoid(x)		
		

		
		

	
