#%%
import matplotlib.pyplot as plt
import numpy as np

# #%%
# #MNIST-SSIM
# y = [2, 4, 8, 16, 24, 32]

# # MNIST
# x1 = [0.40653318, 0.48632136, 0.47485107, 0.43514252, 0.45699447, 0.44944817]
# e1 = [0.07319353, 0.059060536, 0.035698585, 0.014337503, 0.021017369, 0.014016349]

# x2 = [0.37183112, 0.44978923, 0.45205393, 0.43031326, 0.44786534, 0.43671495]
# e2 = [0.096676745, 0.04909963, 0.04860523, 0.021409854, 0.008376499, 0.0208972966]

# plt.errorbar(y, x1, yerr=e1, marker='^', label='pgd')
# plt.errorbar(y, x2, yerr=e2, marker='*', label='our')
# plt.legend(loc='lower right')
# plt.xlabel("Number of particles")
# plt.ylabel("SSIM")
# plt.xticks(y)

# # plt.plot(y, x1, y, x2)
# plt.savefig("figs/"+"MNIST-SSIM.pdf", bbox_inches='tight')
# plt.show()

#%%
# MNIST-SSE
y = [2, 4, 8, 16, 24, 32]

# MNIST
x1 = [1.134149, 1.1349211, 1.0501223, 1.0910736, 1.1010495, 1.1172327]
e1 = [0.24683575, 0.15486543, 0.037362963, 0.034217034, 0.030965818, 0.027173787]

x2 = [1.1764957, 1.1551642, 1.1341361, 1.123179, 1.113113, 1.1449744]
e2 = [0.10881159, 0.078049324, 0.06545849, 0.078965366, 0.027317924, 0.04416823]

plt.errorbar(y, x1, yerr=e1, marker="^", label="pgd")
plt.errorbar(y, x2, yerr=e2, marker="*", label="our")
plt.legend(loc="lower right")
plt.xlabel("Number of particles")
plt.ylabel("Sum Square Error")
plt.xticks(y)

# plt.plot(y, x1, y, x2)
plt.savefig("figs/" + "MNIST-SSE.pdf", bbox_inches="tight")
plt.show()

#%%
# #CIfar10-SSIM
# y = [2, 4, 8, 16, 24, 32]

# #CIFAR10
# x1 = [0.6208552, 0.6246127, 0.62281096, 0.62410015, 0.62415576, 0.6250334]
# e1 = [0.011164783, 0.003579707, 0.0013486657, 0.0018083408, 0.001185644, 0.00067940756]

# x2 = [0.6168374, 0.62377995, 0.61994445, 0.622028, 0.6227106, 0.62376565]
# e2 = [0.00919477, 0.0059341937, 0.0041665314, 0.002820079, 0.0022124897, 0.0009886698]


# plt.errorbar(y, x1, yerr=e1, marker='^', label='pgd')
# plt.errorbar(y, x2, yerr=e2, marker='*', label='our')
# plt.legend(loc='lower right')
# plt.xlabel("Number of particles")
# plt.ylabel("SSIM")
# plt.xticks(y)

# # plt.plot(y, x1, y, x2)
# plt.savefig("figs/"+"CIFAR10-SSIM.pdf", bbox_inches='tight')
# plt.show()

#%%
# CIfar10-SSE
y = [2, 4, 8, 16, 24, 32]

# CIFAR10
x1 = [28.401613, 29.641329, 29.753578, 29.855133, 29.958609, 29.986137]
e1 = [1.0075307, 0.4213126, 0.20767535, 0.32071084, 0.27774778, 0.29934993]

x2 = [28.967817, 29.752203, 30.02474, 30.218542, 30.200647, 30.021591]
e2 = [0.8581285, 1.1987902, 0.34182143, 0.22931886, 0.22050491, 0.10491078]


plt.errorbar(y, x1, yerr=e1, marker="^", label="pgd")
plt.errorbar(y, x2, yerr=e2, marker="*", label="our")
plt.legend(loc="lower right")
plt.xlabel("Number of particles")
plt.ylabel("Sum Square Error")
plt.xticks(y)

# plt.plot(y, x1, y, x2)
plt.savefig("figs/" + "CIFAR10-SSE.pdf", bbox_inches="tight")
plt.show()


#%%
# #CIfar10-Mean Absolute Error
# y = [2, 4, 8, 16, 24, 32]

# #CIFAR10
# x1 = [244.95682, 243.99507, 245.64941, 244.12747, 243.77425, 244.09831]
# e1 = [6.402035, 1.6818892, 1.1335431, 0.66834724, 0.49448806, 0.6755813]

# x2 = []
# e2 = []


# plt.errorbar(y, x1, yerr=e1, marker='^', label='pgd')
# plt.errorbar(y, x2, yerr=e2, marker='*', label='our')
# plt.legend(loc='lower right')
# plt.xlabel("Number of particles")
# plt.ylabel("Mean Absolute Error")
# plt.xticks(y)

# # plt.plot(y, x1, y, x2)
# plt.savefig("figs/"+"CIFAR10-ABS.pdf", bbox_inches='tight')
# plt.show()

# %%
# MNIST-300
x = [1, 2, 4, 8, 16, 24]
y3_base = [0.9536]
y3_svgd = [0.9825, 0.986, 0.9884, 0.9841, 0.9816, 0.985]
y3_vat = [0.9642, 0.9682, 0.9688, 0.9682, 0.9647, 0.9644]
plt.errorbar([1], y3_base, marker="^", label="baseline")
plt.errorbar(x, y3_svgd, marker="*", label="our_vat")
plt.errorbar(x, y3_vat, marker="o", label="vat")
plt.legend(loc="lower right")
plt.xlabel("Number of particles")
plt.ylabel("Accuracy")
plt.ylim(0.95, 0.99)
plt.xticks(x)

# plt.plot(y, x1, y, x2)
plt.savefig("figs/" + "MNIST-SEMI-300.pdf", bbox_inches="tight")
plt.show()


# %%
# MNIST-500
x = [1, 2, 4, 8, 16, 24]
y5_base = [0.9681]
y5_svgd = [0.9881, 0.9893, 0.99, 0.987, 0.9853, 0.9849]
y5_vat = [0.9778, 0.9754, 0.9798, 0.9807, 0.9824, 0.9756]
plt.errorbar([1], y5_base, marker="^", label="baseline")
plt.errorbar(x, y5_svgd, marker="*", label="our_vat")
plt.errorbar(x, y5_vat, marker="o", label="vat")
plt.legend(loc="lower right")
plt.xlabel("Number of particles")
plt.ylabel("Accuracy")
plt.ylim(0.965, 0.991)
plt.xticks(x)

# plt.plot(y, x1, y, x2)
plt.savefig("figs/" + "MNIST-SEMI-500.pdf", bbox_inches="tight")
plt.show()

# %%
# CIFAR10-1000
x = [1, 2, 4, 8]
y_base = [0.7174]
y_svgd = [0.748, 0.755, 0.7777, 0.7762]
y_vat = [0.753, 0.7519, 0.7576, 0.7516]
plt.errorbar([1], y_base, marker="^", label="baseline")
plt.errorbar(x, y_svgd, marker="*", label="our_vat")
plt.errorbar(x, y_vat, marker="o", label="vat")
plt.legend(loc="lower right")
plt.xlabel("Number of particles")
plt.ylabel("Accuracy")
# plt.ylim(0.965, 0.991)
plt.xticks(x)

# plt.plot(y, x1, y, x2)
plt.savefig("figs/" + "CIFAR10-SEMI-1000.pdf", bbox_inches="tight")
plt.show()

# %%
# CIFAR10-4000
x = [1, 2, 4, 8]
y_base = [0.8459]
y_svgd = [0.867, 0.8757, 0.8818, 0.8723]
y_vat = [0.8601, 0.8611, 0.8585, 0.8566]
plt.errorbar([1], y_base, marker="^", label="baseline")
plt.errorbar(x, y_svgd, marker="*", label="our_vat")
plt.errorbar(x, y_vat, marker="o", label="vat")
plt.legend(loc="lower right")
plt.xlabel("Number of particles")
plt.ylabel("Accuracy")
# plt.ylim(0.965, 0.991)
plt.xticks(x)

# plt.plot(y, x1, y, x2)
plt.savefig("figs/" + "CIFAR10-SEMI-4000.pdf", bbox_inches="tight")
plt.show()

# %%
# CIFAR10-4000-VGG
x = [1, 2, 4, 8]
y_base = [0.641]
y_svgd = [0.647, 0.6696, 0.6762, 0.67515]
y_vat = [0.6573, 0.6581, 0.659, 0.6525]
plt.errorbar([1], y_base, marker="^", label="baseline")
plt.errorbar(x, y_svgd, marker="*", label="our_vat")
plt.errorbar(x, y_vat, marker="o", label="vat")
plt.legend(loc="lower right")
plt.xlabel("Number of particles")
plt.ylabel("Accuracy")
# plt.ylim(0.965, 0.991)
plt.xticks(x)

# plt.plot(y, x1, y, x2)
plt.savefig("figs/" + "CIFAR10-SEMI-VGG-4000.pdf", bbox_inches="tight")
plt.show()

# %%
# ROBUST-MNIST
x = [1, 2, 4, 8, 16, 24]
pgd_y_nat = [0.9948, 0.9951, 0.9948, 0.9949, 0.9946, 0.9952]
trades_y_nat = [0.9942, 0.9932, 0.9935, 0.9929, 0.9917, 0.9901]

pgd_y_rob = [0.952, 0.9529, 0.9551, 0.9567, 0.9592, 0.9535]
trades_y_rob = [0.9537, 0.9569, 0.954, 0.9576, 0.9619, 0.9613]

fig, ax1 = plt.subplots()

ax1.errorbar(x, pgd_y_nat, marker="^", label="pgd_natural_acc")
ax1.errorbar(x, trades_y_nat, marker="^", label="trades_natural_acc")
ax1.set_xlabel("Number of particles")
ax1.set_ylabel("Natural accuracy")
plt.legend(loc="center right")

ax2 = ax1.twinx()

ax2.errorbar(x, pgd_y_rob, marker="o", label="pgd_robust_acc")
ax2.errorbar(x, trades_y_rob, marker="o", label="trades_robust_acc")

ax2.set_ylabel("Robust accuracy")
plt.legend(loc="center left")
plt.xticks(x)

# plt.plot(y, x1, y, x2)
plt.savefig("figs/" + "ROBUST-MNIST.pdf", bbox_inches="tight")
plt.show()
# %%
# ROBUST-CIFAR10-CNN
x = [1, 2, 4, 8]
pgd_y_nat = [0.7569, 0.7597, 0.7519, 0.7479]
trades_y_nat = [0.6543, 0.6595, 0.6459, 0.6458]

pgd_y_rob = [0.3349, 0.3394, 0.345, 0.3423]
trades_y_rob = [0.3475, 0.3512, 0.3586, 0.3513]

fig, ax1 = plt.subplots()

ax1.errorbar(x, pgd_y_nat, marker="^", label="pgd_natural_acc")
ax1.errorbar(x, trades_y_nat, marker="^", label="trades_natural_acc")
ax1.set_xlabel("Number of particles")
ax1.set_ylabel("Natural accuracy")
plt.legend(loc="center right")

ax2 = ax1.twinx()

ax2.errorbar(x, pgd_y_rob, marker="o", label="pgd_robust_acc")
ax2.errorbar(x, trades_y_rob, marker="o", label="trades_robust_acc")

ax2.set_ylabel("Robust accuracy")
plt.legend(loc="center left")
plt.xticks(x)

# plt.plot(y, x1, y, x2)
plt.savefig("figs/" + "ROBUST-CIFAR10-CNN.pdf", bbox_inches="tight")
plt.show()
# %%
# ROBUST-CIFAR10-Resnet18
x = [1, 2, 4, 8]
pgd_y_nat = [0.8528, 0.8501, 0.8574, 0.8559]
trades_y_nat = [0.8339, 0.8038, 0.778, 0.7417]

pgd_y_rob = [0.4515, 0.4562, 0.4643, 0.4559]
trades_y_rob = [0.5259, 0.5318, 0.5389, 0.5244]

fig, ax1 = plt.subplots()

ax1.errorbar(x, pgd_y_nat, marker="^", label="pgd_natural_acc")
ax1.errorbar(x, trades_y_nat, marker="^", label="trades_natural_acc")
ax1.set_xlabel("Number of particles")
ax1.set_ylabel("Natural accuracy")
plt.legend(loc="center right")

ax2 = ax1.twinx()

ax2.errorbar(x, pgd_y_rob, marker="o", label="pgd_robust_acc")
ax2.errorbar(x, trades_y_rob, marker="o", label="trades_robust_acc")

ax2.set_ylabel("Robust accuracy")
plt.legend(loc="center left")
plt.xticks(x)

# plt.plot(y, x1, y, x2)
plt.savefig("figs/" + "ROBUST-CIFAR10-RESNET18.pdf", bbox_inches="tight")
plt.show()

# %%
# # Bar Chart MNIST
# N = 2
# pgd = (0.9503, 0.952)
# pgd_16 = (0.9511, 0.9592)
# trades = (0.9565, 0.9537)
# trades_16 = (0.9565, 0.9619)

# ind = np.arange(N)
# width = 0.2
# plt.bar(ind, pgd, width, label='pgd')
# plt.bar(ind + width, pgd_16, width, label='pgd_16')
# plt.bar(ind + 2*width, trades, width, label='trades')
# plt.bar(ind + 3*width, trades_16, width, label='trades_16')

# plt.ylabel('Robust accuracy')
# plt.title('PGD-ATTACK MNIST')

# plt.xticks(ind + 4*width / 2, ('Baseline', 'Our'))
# plt.legend(loc='best')
# plt.ylim(0.945,0.965)
# plt.savefig("figs/"+"PGD-ATTACK MNIST-ROB.pdf", bbox_inches='tight')
# plt.show()

# # natural
# pgd = (0.9947, 0.9948)
# pgd_16 = (0.9947, 0.9946)
# trades = (0.9942, 0.9942)
# trades_16 = (0.993, 0.9917)

# ind = np.arange(N)
# width = 0.2
# plt.bar(ind, pgd, width, label='pgd')
# plt.bar(ind + width, pgd_16, width, label='pgd_16')
# plt.bar(ind + 2*width, trades, width, label='trades')
# plt.bar(ind + 3*width, trades_16, width, label='trades_16')

# plt.ylabel('Natural accuracy')
# plt.title('PGD-ATTACK MNIST')

# plt.xticks(ind + 4*width / 2, ('Baseline', 'Our'))
# plt.legend(loc='best')
# plt.ylim(0.991, 0.9962)
# plt.savefig("figs/"+"PGD-ATTACK MNIST-NAT.pdf", bbox_inches='tight')
# plt.show()

# %%
# # %%
# # Bar Chart CIFAR10
# N = 2
# pgd = (0.4556, 0.4515)
# pgd_4 = (0.4562, 0.4634)
# trades = (0.5251, 0.5259)
# trades_4 = (0.5268, 0.5389)

# ind = np.arange(N)
# width = 0.2
# plt.bar(ind, pgd, width, label='pgd')
# plt.bar(ind + width, pgd_4, width, label='pgd_4')
# plt.bar(ind + 2*width, trades, width, label='trades')
# plt.bar(ind + 3*width, trades_4, width, label='trades_4')

# plt.ylabel('Robust accuracy')
# plt.title('PGD-ATTACK CIFAR10')

# plt.xticks(ind + 4*width / 2, ('Baseline', 'Our'))
# plt.legend(loc='best')
# plt.ylim(0.44,0.56)
# plt.savefig("figs/"+"PGD-ATTACK CIFAR10-ROB.pdf", bbox_inches='tight')
# plt.show()

# # natural
# pgd = (0.8499, 0.8528)
# pgd_4 = (0.8488, 0.8574)
# trades = (0.8207, 0.8339)
# trades_4 = (0.7499, 0.7417)

# ind = np.arange(N)
# width = 0.2
# plt.bar(ind, pgd, width, label='pgd')
# plt.bar(ind + width, pgd_4, width, label='pgd_4')
# plt.bar(ind + 2*width, trades, width, label='trades')
# plt.bar(ind + 3*width, trades_4, width, label='trades_4')

# plt.ylabel('Natural accuracy')
# plt.title('PGD-ATTACK CIFAR10')

# plt.xticks(ind + 4*width / 2, ('Baseline', 'Our'))
# plt.legend(loc='best')
# plt.ylim(0.73, 0.89)
# plt.savefig("figs/"+"PGD-ATTACK CIFAR10-NAT.pdf", bbox_inches='tight')
# plt.show()


# %%
# MNIST-pgd
x = [1, 2, 4, 8, 16, 24]
y_svgd = [0.952, 0.9529, 0.9551, 0.9567, 0.9592, 0.9535]
y_vat = [0.9499, 0.9503, 0.9511, 0.9505, 0.9513, 0.9489]
plt.errorbar(x, y_svgd, marker="*", label="our_pgd")
plt.errorbar(x, y_vat, marker="o", label="pgd")
plt.legend(loc="lower right")
plt.xlabel("Number of particles")
plt.ylabel("Robust accuracy")
# plt.ylim(0.965, 0.991)
plt.xticks(x)

# plt.plot(y, x1, y, x2)
plt.savefig("figs/" + "MNIST-PGD-ROB.pdf", bbox_inches="tight")
plt.show()

x = [1, 2, 4, 8, 16, 24]
y_svgd = [0.9948, 0.9951, 0.9948, 0.9949, 0.9946, 0.9947]
y_vat = [0.9951, 0.9947, 0.9947, 0.9944, 0.9945, 0.9944]
plt.errorbar(x, y_svgd, marker="*", label="our_pgd")
plt.errorbar(x, y_vat, marker="o", label="pgd")
plt.legend(loc="lower right")
plt.xlabel("Number of particles")
plt.ylabel("Natural accuracy")
# plt.ylim(0.965, 0.991)
plt.xticks(x)

# plt.plot(y, x1, y, x2)
plt.savefig("figs/" + "MNIST-PGD-NAT.pdf", bbox_inches="tight")
plt.show()

# %%
# MNIST-trades
x = [1, 2, 4, 8, 16, 24]
y_svgd = [0.9537, 0.9569, 0.954, 0.9576, 0.9619, 0.9613]
y_vat = [0.9565, 0.9567, 0.9565, 0.9568, 0.9552, 0.955]
plt.errorbar(x, y_svgd, marker="*", label="our_trades")
plt.errorbar(x, y_vat, marker="o", label="trades")
plt.legend(loc="lower right")
plt.xlabel("Number of particles")
plt.ylabel("Robust accuracy")
# plt.ylim(0.965, 0.991)
plt.xticks(x)

# plt.plot(y, x1, y, x2)
plt.savefig("figs/" + "MNIST-TRADES-ROB.pdf", bbox_inches="tight")
plt.show()

x = [1, 2, 4, 8, 16, 24]
y_svgd = [0.9942, 0.9932, 0.9935, 0.9929, 0.9917, 0.9901]
y_vat = [0.9942, 0.9942, 0.993, 0.9926, 0.9921, 0.9901]
plt.errorbar(x, y_svgd, marker="*", label="our_trades")
plt.errorbar(x, y_vat, marker="o", label="trades")
plt.legend(loc="lower right")
plt.xlabel("Number of particles")
plt.ylabel("Natural accuracy")
# plt.ylim(0.965, 0.991)
plt.xticks(x)

# plt.plot(y, x1, y, x2)
plt.savefig("figs/" + "MNIST-TRADES-NAT.pdf", bbox_inches="tight")
plt.show()
# %%
# CIFAR10-pgd
x = [1, 2, 4, 8]
y_svgd = [0.4515, 0.4562, 0.4715, 0.4559]
y_vat = [0.4556, 0.4562, 0.4572, 0.4538]
plt.errorbar(x, y_svgd, marker="*", label="our_pgd")
plt.errorbar(x, y_vat, marker="o", label="pgd")
plt.legend(loc="lower right")
plt.xlabel("Number of particles")
plt.ylabel("Robust accuracy")
# plt.ylim(0.965, 0.991)
plt.xticks(x)

# plt.plot(y, x1, y, x2)
plt.savefig("figs/" + "CIFAR10-PGD-ROB.pdf", bbox_inches="tight")
plt.show()

x = [1, 2, 4, 8]
y_svgd = [0.8528, 0.8501, 0.8574, 0.8559]
y_vat = [0.8499, 0.8498, 0.8488, 0.8432]
plt.errorbar(x, y_svgd, marker="*", label="our_pgd")
plt.errorbar(x, y_vat, marker="o", label="pgd")
plt.legend(loc="lower right")
plt.xlabel("Number of particles")
plt.ylabel("Natural accuracy")
# plt.ylim(0.965, 0.991)
plt.xticks(x)

# plt.plot(y, x1, y, x2)
plt.savefig("figs/" + "CIFAR10-PGD-NAT.pdf", bbox_inches="tight")
plt.show()

# %%
# CIFAR10-trades
x = [1, 2, 4, 8]
y_svgd = [0.5259, 0.5318, 0.5389, 0.5244]
y_vat = [0.5251, 0.5257, 0.5268, 0.5257]
plt.errorbar(x, y_svgd, marker="*", label="our_trades")
plt.errorbar(x, y_vat, marker="o", label="trades")
plt.legend(loc="lower right")
plt.xlabel("Number of particles")
plt.ylabel("Robust accuracy")
# plt.ylim(0.965, 0.991)
plt.xticks(x)

# plt.plot(y, x1, y, x2)
plt.savefig("figs/" + "CIFAR10-TRADES-ROB.pdf", bbox_inches="tight")
plt.show()

x = [1, 2, 4, 8]
y_svgd = [0.8339, 0.8038, 0.778, 0.7417]
y_vat = [0.8207, 0.7802, 0.7499, 0.7402]
plt.errorbar(x, y_svgd, marker="*", label="our_trades")
plt.errorbar(x, y_vat, marker="o", label="trades")
plt.legend(loc="lower right")
plt.xlabel("Number of particles")
plt.ylabel("Natural accuracy")
# plt.ylim(0.965, 0.991)
plt.xticks(x)

# plt.plot(y, x1, y, x2)
plt.savefig("figs/" + "CIFAR10-TRADES-NAT.pdf", bbox_inches="tight")
plt.show()
# %%
# Domain adaptation SVHN -> MNIST
x = [1, 2, 4, 8, 16]
y_svgd = [0.956, 0.974, 0.981, 0.986, 0.982]
y_vat = [0.954, 0.957, 0.957, 0.964, 0.968]
plt.errorbar(x, y_svgd, marker="*", label="our_vada")
plt.errorbar(x, y_vat, marker="o", label="vada")
plt.legend(loc="lower right")
plt.xlabel("Number of particles")
plt.ylabel("Accuracy")
# plt.ylim(0.965, 0.991)
plt.xticks(x)

# plt.plot(y, x1, y, x2)
plt.savefig("figs/" + "SVHN_MNIST_DA.pdf", bbox_inches="tight")
plt.show()

#%%
# Domain adaptation  MNIST -> SVHN
x = [1, 2, 4, 8, 16]
y_svgd = [0.72, 0.724, 0.732, 0.737, 0.727]
y_vat = [0.727, 0.729, 0.73, 0.733, 0.72]
plt.errorbar(x, y_svgd, marker="*", label="our_vada")
plt.errorbar(x, y_vat, marker="o", label="vada")
plt.legend(loc="lower right")
plt.xlabel("Number of particles")
plt.ylabel("Accuracy")
# plt.ylim(0.965, 0.991)
plt.xticks(x)

# plt.plot(y, x1, y, x2)
plt.savefig("figs/" + "MNIST_SVHN_DA.pdf", bbox_inches="tight")
plt.show()
# %%
