from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

class WHint(nn.Module):
	def __init__(self):
		super(WHint, self).__init__()
	def forward(self, fm_s, fm_t):
		#fm_s 2*128*512
		weight, _ = torch.max(fm_t, 1)#2*512
		weight_soft = torch.nn.functional.softmax(weight, 1).unsqueeze(1)#2*512
		fm_s = fm_s * weight_soft * weight_soft.shape[2]
		fm_t = fm_t * weight_soft * weight_soft.shape[2]
		loss = F.mse_loss(fm_s, fm_t)
		return loss

class Hint(nn.Module):
	'''
	FitNets: Hints for Thin Deep Nets
	https://arxiv.org/pdf/1412.6550.pdf
	'''
	def __init__(self):
		super(Hint, self).__init__()

	def forward(self, fm_s, fm_t):
		loss = F.mse_loss(fm_s, fm_t)

		return loss