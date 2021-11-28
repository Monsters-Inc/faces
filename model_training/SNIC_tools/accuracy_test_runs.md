#
# Model architecture - 5 runs each
#

# 16, 32, (64, 64), (128, 128), (128, 128) - Dropout: 0.2
- 0.8839694619178772 4x FC-64
- 0.7932824492454529 3x FC-64 (3 were ~90%, 2 were ~60%)
- 0.8387786269187927 2x FC-64 (1 was ~60%)
- 0.894656491279602 1x FC-64 (stable through all runs)

# 16, 32, (64, 64), (128, 128), (256, 256) - Dropout = 0.2
- 0.8873282432556152 (Really stable throughout)

# 16, 32, (64, 64), (128, 128), (256, 512) - Dropout: 0.2
- 0.7926717400550842 (Really unstable)

# 16, 32, 64, 128, 256, 512 - Dropout: 0.2 - With dropout before output
- 0.8986259579658509 (Pretty stable)

# (16, 16), (32, 32), (64, 64), (128, 128), (256, 256) - Dropout: 0.2
- 0.7807633519172669 (First three were around 0.90, but the two last were like ~0.5-0.6)

# 16, 32, 64, (128, 128), (256, 256) - Dropout: 0.2
- 0.9032061100006104
- 0.9010686993598938

# 16, 32, 64, (128, 128), (256, 256), (512, 512) - Dropout: 0.2
- 0.7450381875038147

# 16, 32, 64, (128, 128), (256, 512) - Dropout: 0.2
- 0.900458014011383
- 0.9001526832580566

# 16, 32, 64, (128, 128), (256, 512) - Dropout: 0.4
- 0.8812213778495789

# 16, 32, 64, (128, 128), (256, 512) - Dropout: 0.2 - Added dropout before output: 0.2
- 0.9029007673263549

# 16, 32, 64, (128, 128), (256, 512) - Dropout: 0.2 - Added dropout before output: 0.8
- 0.8406106948852539

# 16, 32, 64, (128, 128), (256, 512) - Dropout: 0.2 - Added dropout before output: 0.1 - Best result
- 0.9047328233718872

#
# Preprocessing - 10 runs each
#

# FINAL | Color | Average of 10 runs
- 0.8954198598861695

# FINAL | Grayscale | Average of 10 runs
- 0.8934351146221161

# FINAL | HE | Average of 10 runs
- 0.9160305321216583
- 0.9206106901168823 - 30 runs

# FINAL | CLAHE | Average of 10 runs
- 0.9022900879383087

# FINAL | Canny Edges | Average of 10 runs
- 0.8574045717716217

# FINAL | BGR | Average of 10 runs
- 0.8696183204650879 - May be a run with only 1 channel
- 0.8697709977626801 - Correct number of channels