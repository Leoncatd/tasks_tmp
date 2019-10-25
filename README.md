# Pretrain Task: Image Inpainting and Language Completion

## Model architecture

``` 
        IMAGE                            TOKENS
0      256x256, 3                         30x1       0
---- conv3x3 stride2 ----------------- BertLayer -----
1      128x128, 64                        30x768     1
---- conv3x3 stride2 ----------------- BertLayer -----
2      64x64, 128                         30x768     2
---- conv3x3 stride2 ----------------- BertLayer -----
3      32x32, 256          fusion         30x768     3
---- conv3x3 stride2 ----------------- BertLayer -----
4      16x16, 512          fusion         30x768     4
---- conv3x3 stride2 ----------------- BertLayer -----
5      8x8, 512            fusion         30x768     5
---- conv3x3 stride2 ----------------- BertLayer -----
6      4x4, 512            fusion         30x768     6
---- conv3x3 stride2 ----------------- BertLayer -----
7      2x2, 512            fusion         30x768     7
---- dconv3x3 stride2 ---------------- BertLayer -----
8      4x4, 512            fusion         30x768     8
---- dconv3x3 stride2 ---------------- BertLayer -----
9      8x8, 512            fusion         30x768     9
---- dconv3x3 stride2 ---------------- BertLayer -----
10     16x16, 512          fusion         30x768    10
---- dconv3x3 stride2 ---------------- BertLayer -----
11     32x32, 256          fusion         30x768    11
---- dconv3x3 stride2 ---------------- BertLayer -----
12     64x64, 128                         30x768    12
---- dconv3x3 stride2 ---------------- BertLayer -----
13     128x128, 64                        30x768    13
---- dconv3x3 stride2 ---------------- BertLayer -----
14     256x256 3                          30x768    14
```

fusion
```
32x32x256                                        30x768
 |----.-----.--------.            .-------.-----.----|
 |    |  conv1x1  conv1x1      linear   linear  |    |
 |    |     |        |            |       |     |    |
 |    | 32x32x512 32x32x256    30x768   25x1536 |    |
 |    |     |        |            |       |     |    |
 |    |     |        `--multiply--'       |     |    |
 |    |     |              |              |     |    |
 |    |     |         1024*30*4*24        |     |    |
 |    |     |              |              |     |    |
 |    |     `---multiply---^---multiply---'     |    |
 |    |            |               |            |    |
 |    |         30x1536        32x32x512        |    |
 |    |            |               |            |    |
 |    |          linear         conv1x1         |    |
 |    |            |               |            |    |
 |    |         30x1536        32x32x512        |    |
 |    |            '---------------|---------concate |
 | concate-------------------------'            |    |
 |    |                                         |    |
 | conv1x1                                    linear |
 |    |                                         |    |
add---'                                         '---add
 |                                                   |
32x32x256                                        30x768
```

# Partially auto-regressive Image-to-Text generation

## Model architecture
```
        IMAGE
0      256x256, 3
---- conv3x3 stride2
1      128x128, 64
---- conv3x3 stride2                   30x1
2      64x64, 128                   BertEmbedding
---- conv3x3 stride2                   30x768
       32x32, 256 ----------------> ShowAttendTell 
3          |             fusion        30x768    3 -> LSTMHead
---- conv3x3 stride2 -------------- BertLayer ----
4      16x16, 512        fusion        30x768    4
---- conv3x3 stride2 -------------- BertLayer ----
5      8x8, 512          fusion        30x768    5
---- conv3x3 stride2 -------------- BertLayer ----
6      4x4, 512          fusion        30x768    6
---- conv3x3 stride2 -------------- BertLayer ----
7      2x2, 512          fusion        30x768    7 -> LSTMHead
---- dconv3x3 stride2 ------------- BertLayer ----
8      4x4, 512          fusion        30x768    8
---- dconv3x3 stride2 ------------- BertLayer ----
9      8x8, 512          fusion        30x768    9
---- dconv3x3 stride2 ------------- BertLayer ----
10     16x16, 512        fusion        30x768   10
---- dconv3x3 stride2 ------------- BertLayer ----
11     32x32, 256        fusion        30x768   11 -> LSTMHead
                                    BertLayer ----
                                       30x768   12
                                    BertLayer ----
                                       30x768   13
                                    BertLayer ----
                                       30x768   14 -> LSTMHead
```

ShowAttendTell
```
```

LSTMHead
```
```

# Progressive Text-to-Image generation

## Model architecture
```
                                                  TOKENS
                                                   30x1      0
                                                BertLayer ----
                                                   30x768    1
                                                BertLayer ----
                                                   30x768    2
                                                BertLayer ----
            3      32x32, 256 <----- fusion ------ 30x768    3
            ---- conv3x3 stride2 -------------- BertLayer ----
            4      16x16, 512        fusion        30x768    4
  NOISE     ---- conv3x3 stride2 -------------- BertLayer ----
 2x2, 32    5      8x8, 512          fusion        30x768    5
 conv3x3    ---- conv3x3 stride2 -------------- BertLayer ----
2x2, 128    6      4x4, 512          fusion        30x768    6
 conv3x3    ---- conv3x3 stride2 -------------- BertLayer ----
2x2, 512 -> 7      2x2, 512          fusion        30x768    7
            ---- dconv3x3 stride2 ------------- BertLayer ----
            8      4x4, 512          fusion        30x768    8
            ---- dconv3x3 stride2 ------------- BertLayer ----
            9      8x8, 512          fusion        30x768    9
            ---- dconv3x3 stride2 ------------- BertLayer ----
            10     16x16, 512        fusion        30x768   10
            ---- dconv3x3 stride2 ------------- BertLayer ----
            11     32x32, 256        fusion        30x768   11
            ---- dconv3x3 stride2 -------------      |
 conv1x1 <- 12     64x64, 128        fusion        30x768
64x64, 3              |                              |
  D_64           dconv3x3 stride2                    |
                   Basic Block                       |
                   128x128, 64 <---- fusion ------ 30x768
                   Basic Block                       |
 conv1x1 <-------- 128x128, 64                       |
128x128, 3            |                              |
  D_128          dconv3x3 stride2                    |
                   Basic Block                       |
                   256x256, 32 <---- fusion ------ 30x768
                   Basic Block                
 conv1x1 <-------- 256x256, 32
256x256, 3
  D_256
```

D_64
```
```
