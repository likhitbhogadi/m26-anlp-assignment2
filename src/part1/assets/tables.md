## Parameters

| variant | total | active | FFN total | FFN active | non-emb total | non-emb active |
|---|---|---|---|---|---|---|
| dense | 35.27M | 35.27M | 12.58M | 12.58M | 18.88M | 18.88M |
| moe_4e_top1 | 35.28M | 25.84M | 12.60M | 3.16M | 18.89M | 9.46M |
| moe_4e_top2 | 35.28M | 28.99M | 12.60M | 6.30M | 18.89M | 12.60M |
| moe_shared1_3e_top1 | 35.27M | 28.98M | 12.59M | 6.30M | 18.89M | 12.60M |
| moe_4e_top2_active | 47.86M | 35.28M | 25.18M | 12.60M | 31.48M | 18.89M |

## Test results

| variant | PPL vi→en | PPL ja→en | PPL both | BLEU vi→en | BLEU ja→en | BLEU both |
|---|---|---|---|---|---|---|
| dense | 8.26 | 13.36 | 10.51 | 34.01 | 24.11 | 29.09 |
| moe_4e_top1 | 9.55 | 15.98 | 12.35 | 31.62 | 21.87 | 26.77 |
| moe_4e_top2 | 8.75 | 14.48 | 11.26 | 32.93 | 22.73 | 27.79 |
| moe_shared1_3e_top1 | 8.64 | 14.07 | 11.02 | 32.99 | 23.07 | 28.05 |
| moe_4e_top2_active | 8.23 | 13.45 | 10.52 | 34.37 | 24.30 | 29.36 |

## Expert usage (all test tokens; mean over layers, [min–max] across layers)

| variant | load CV | busiest ÷ uniform | normalised entropy | JS vi/ja src | JS vi/ja tgt |
|---|---|---|---|---|---|
| moe_4e_top1 | 0.043 [0.010–0.071] | 1.052 [1.014–1.093] | 0.999 [0.998–1.000] | 0.005 [0.002–0.008] | 0.000 [0.000–0.001] |
| moe_4e_top2 | 0.060 [0.019–0.131] | 1.092 [1.021–1.223] | 0.998 [0.994–1.000] | 0.007 [0.001–0.024] | 0.000 [0.000–0.000] |
| moe_shared1_3e_top1 | 0.044 [0.017–0.083] | 1.054 [1.019–1.117] | 0.999 [0.997–1.000] | 0.013 [0.001–0.056] | 0.002 [0.000–0.006] |
| moe_4e_top2_active | 0.052 [0.018–0.138] | 1.081 [1.021–1.238] | 0.999 [0.993–1.000] | 0.004 [0.000–0.009] | 0.000 [0.000–0.001] |

## Tokens most consistently routed to each expert (test set, tokens with ≥200 occurrences)

purity = share of the token's occurrences routed to that expert (uniform routing: top_k / n_experts).

**moe_4e_top1**

| layer | expert | top tokens (purity) |
|---|---|---|
| L0 | E0 | `▁mà` (1.00), `そして` (1.00), `との` (1.00), `▁私は` (1.00), `まで` (1.00), `「` (1.00), `▁それは` (1.00), `するために` (1.00) |
| L0 | E1 | `▁just` (1.00), `▁Why` (1.00), `▁company` (1.00), `▁products` (1.00), `▁way` (1.00), `▁offer` (1.00), `▁his` (1.00), `▁delivery` (1.00) |
| L0 | E2 | `▁địa` (1.00), `▁phụ` (1.00), `▁quả` (1.00), `▁thay` (1.00), `▁tiền` (1.00), `▁xe` (1.00), `▁đánh` (1.00), `▁hóa` (1.00) |
| L0 | E3 | `▁after` (1.00), `▁Number:` (1.00), `▁through` (1.00), `▁between` (1.00), `<2en>` (1.00), `▁during` (1.00), `▁around` (1.00), `▁into` (1.00) |
| L5 | E0 | `▁same` (1.00), `▁business` (1.00), `▁payment` (1.00), `▁customer` (1.00), `▁who` (1.00), `▁they` (1.00), `▁production` (0.99), `▁most` (0.99) |
| L5 | E1 | `<bos>` (1.00), `<vi>` (1.00), `▁tham` (1.00), `▁tải` (1.00), `▁get` (1.00), `▁want` (1.00), `▁cập` (0.99), `▁đăng` (0.99) |
| L5 | E2 | `▁Number:` (1.00), `▁tôi.` (1.00), `しました。` (1.00), `<2en>` (1.00), `します。` (1.00), `▁•` (1.00), `しています。` (1.00), `することができます。` (1.00) |
| L5 | E3 | `▁mặt` (1.00), `▁Sh` (1.00), `▁y` (1.00), `▁Th` (1.00), `▁Máy` (1.00), `▁j` (1.00), `▁tr` (1.00), `▁ch` (1.00) |

**moe_4e_top2**

| layer | expert | top tokens (purity) |
|---|---|---|
| L0 | E0 | `多くの` (1.00), `▁control` (1.00), `▁business` (1.00), `▁chuyên` (1.00), `▁system` (1.00), `80` (1.00), `新しい` (1.00), `▁Video` (1.00) |
| L0 | E1 | `▁nơi` (1.00), `▁bài` (1.00), `▁Điều` (1.00), `▁hiệu` (1.00), `▁quyền` (1.00), `▁rộng` (1.00), `者` (1.00), `▁địa` (1.00) |
| L0 | E2 | `の` (1.00), `;` (1.00), `▁St` (1.00), `▁that` (1.00), `▁as` (1.00), `...` (1.00), `▁mọi` (1.00), `▁its` (1.00) |
| L0 | E3 | `▁always` (1.00), `▁Contact` (1.00), `り` (1.00), `▁have` (1.00), `▁hãy` (1.00), `em` (1.00), `ap` (1.00), `um` (1.00) |
| L5 | E0 | `▁also` (1.00), `▁factory` (1.00), `▁account` (1.00), `▁sample` (1.00), `▁always` (1.00), `▁better` (1.00), `▁most` (1.00), `▁they` (1.00) |
| L5 | E1 | `<bos>` (1.00), `お` (1.00), `▁lựa` (1.00), `▁khả` (1.00), `り` (1.00), `▁hãy` (1.00), `▁【公式サイト】` (1.00), `▁phụ` (1.00) |
| L5 | E2 | `▁English` (1.00), `▁2020` (1.00), `▁etc.` (1.00), `▁United` (1.00), `▁China` (1.00), `できます。` (1.00), `▁2017` (1.00), `することができます。` (1.00) |
| L5 | E3 | `▁em` (1.00), `ル` (1.00), `ン` (1.00), `▁Làm` (1.00), `-s` (1.00), `▁đối` (1.00), `▁Re` (1.00), `▁văn` (1.00) |

**moe_shared1_3e_top1**

| layer | expert | top tokens (purity) |
|---|---|---|
| L0 | E0 | `▁support` (1.00), `▁nối` (1.00), `▁khiển` (1.00), `▁Khách` (1.00), `▁nơi` (1.00), `▁bài` (1.00), `▁mật` (1.00), `▁mở` (1.00) |
| L0 | E1 | `の` (1.00), `することができます。` (1.00), `▁nhưng` (1.00), `それは` (1.00), `▁would` (1.00), `▁đang` (1.00), `▁đã` (1.00), `する` (1.00) |
| L0 | E2 | `▁1)` (1.00), `▁2016` (1.00), `▁Official` (1.00), `▁Our` (1.00), `▁Q:` (1.00), `▁Và` (1.00), `▁2)` (1.00), `▁2.` (1.00) |
| L5 | E0 | `<bos>` (1.00), `<vi>` (1.00), `<ja>` (1.00), `▁cuộc` (1.00), `▁Đ` (0.96), `▁giường` (0.96), `▁Thời` (0.94), `▁Người` (0.94) |
| L5 | E1 | `▁thêm` (0.99), `▁tuyệt` (0.98), `▁trả` (0.98), `▁nhanh` (0.98), `▁đăng` (0.98), `▁tùy` (0.98), `▁dis` (0.97), `▁sp` (0.97) |
| L5 | E2 | `▁【公式サイト】` (1.00), `▁Please` (0.99), `▁will` (0.98), `▁Site` (0.98), `▁There` (0.98), `▁can` (0.98), `▁could` (0.98), `▁who` (0.98) |

**moe_4e_top2_active**

| layer | expert | top tokens (purity) |
|---|---|---|
| L0 | E0 | `▁where` (1.00), `▁what` (1.00), `er.` (1.00), `▁0` (1.00), `▁our` (1.00), `ors` (1.00), `▁how` (1.00), `▁at` (1.00) |
| L0 | E1 | `▁Số` (1.00), `国` (1.00), `ist` (1.00), `▁information` (1.00), `▁version` (1.00), `▁văn` (1.00), `▁sure` (1.00), `▁Site` (1.00) |
| L0 | E2 | `▁no` (1.00), `▁Sh` (1.00), `▁tác` (1.00), `▁vật` (1.00), `▁doanh` (1.00), `▁Chúng` (1.00), `▁sạn` (1.00), `▁Al` (1.00) |
| L0 | E3 | `▁O` (1.00), `st` (1.00), `▁K` (1.00), `▁V` (1.00), `は、` (1.00), `▁re` (1.00), `▁offer` (1.00), `▁E` (1.00) |
| L5 | E0 | `<bos>` (1.00), `い` (1.00), `▁Máy` (1.00), `▁Người` (1.00), `ン` (1.00), `性` (1.00), `▁sự` (1.00), `▁khu` (1.00) |
| L5 | E1 | `▁nhiều` (1.00), `▁あなたは` (1.00), `多くの` (1.00), `私達の` (1.00), `▁mỗi` (1.00), `▁nhau` (1.00), `▁tương` (1.00), `▁hơn` (1.00) |
| L5 | E2 | `<bos>` (1.00), `▁people` (1.00), `<ja>` (1.00), `▁Nếu` (1.00), `▁well` (1.00), `▁app` (1.00), `▁factory` (1.00), `▁products` (1.00) |
| L5 | E3 | `▁low` (1.00), `▁Our` (1.00), `▁Official` (1.00), `▁you.` (1.00), `ly.` (1.00), `▁steel` (0.99), `▁his` (0.99), `ical` (0.99) |


## Sample translations (same random short test sentences for every variant)

**[vi→en, test #6673]** Mẫu Các mẫu được tự do.  
*reference:* Samples Samples are free.

| variant | hypothesis | sentence BLEU |
|---|---|---|
| dense | Sample Samples are free. | 66.9 |
| moe_4e_top1 | Sample sample is free. | 21.4 |
| moe_4e_top2 | Sample sample is free. | 21.4 |
| moe_shared1_3e_top1 | Sample samples are free. | 39.8 |
| moe_4e_top2_active | Sample models are freely. | 12.7 |

**[vi→en, test #12649]** Khối lượng chương trình 120 chương trình, tối đa. 100 dòng trong một chương trình, tối đa 999 vòng lặp.  
*reference:* Program volume 120 program, max. 100 line in one program, max 999 loops.

| variant | hypothesis | sentence BLEU |
|---|---|---|
| dense | The capacity of the program 120 programs, maximum. 100 lines in a program , up to 999 loop. | 6.5 |
| moe_4e_top1 | The program volume 120 programs, maximum maximum. 100 lines in a program for 999 loops. | 13.2 |
| moe_4e_top2 | The current volume of 120 programs, the maximum. 100 lines in one programs, up to 999xes. | 6.6 |
| moe_shared1_3e_top1 | The program volume of 120 programs, maximum. 100 lines in one programs, maximum 999 rimes. | 7.4 |
| moe_4e_top2_active | Program volume 120 programs, up to 100 lines in one program , up to 999 times of repeated . | 19.4 |

**[vi→en, test #21110]** Phần mềm VPN nào là tốt nhất của năm 2019?  
*reference:* What's the best VPN service in 2020?

| variant | hypothesis | sentence BLEU |
|---|---|---|
| dense | What is the best VPN of 2019? | 22.1 |
| moe_4e_top1 | What VPN software is the best of 2019? | 11.3 |
| moe_4e_top2 | What is the best VPN software? | 22.8 |
| moe_shared1_3e_top1 | What VPN software is the best of 2019? | 11.3 |
| moe_4e_top2_active | What VPN Software is Best of 2019? | 6.6 |

**[vi→en, test #15784]** Và chúng tôi cũng có thể cung cấp nền tảng tùy chỉnh để đáp ứng các yêu cầu khác nhau từ khách hàng.  
*reference:* And we can also supply customized platform to meet different requirements from customers.

| variant | hypothesis | sentence BLEU |
|---|---|---|
| dense | And we can also provide custom platform to meet different requirements from customers. | 70.0 |
| moe_4e_top1 | And we can also provide custom platform to meet different requirements from customers. | 70.0 |
| moe_4e_top2 | And we can also provide custom platform to meet different requirements from customers. | 70.0 |
| moe_shared1_3e_top1 | And we can also provide custom platform to meet different requirements from customers. | 70.0 |
| moe_4e_top2_active | And we can also provide custom backgrounds to meet different requirements from customers. | 61.6 |

**[ja→en, test #407]** スパイカーC8スパイダーとオランダの警察  
*reference:* Spyker C8 Spyder and the Dutch police

| variant | hypothesis | sentence BLEU |
|---|---|---|
| dense | Spiders C8 Spider and the Hands of The Netherlands Police | 9.3 |
| moe_4e_top1 | Spiders and the Police of Spiders and the Netherlands Police | 8.4 |
| moe_4e_top2 | Spiner and Dutch police for the Dutch police | 22.1 |
| moe_shared1_3e_top1 | Spiders C8 Spiders and the Netherlands Police | 14.5 |
| moe_4e_top2_active | Spacers C8 Spiders and the Netherlands Police | 14.5 |

**[ja→en, test #20183]** 100%の満足度を保証  
*reference:* ●100% Satisfaction Guarantee

| variant | hypothesis | sentence BLEU |
|---|---|---|
| dense | 100% satisfaction guarantee | 16.0 |
| moe_4e_top1 | 100% satisfaction satisfaction | 16.0 |
| moe_4e_top2 | 100% satisfaction guarantee | 16.0 |
| moe_shared1_3e_top1 | 100% satisfaction guarantee. | 10.7 |
| moe_4e_top2_active | 100% satisfaction guarantee | 16.0 |

**[ja→en, test #4398]** 秘密のサンタへの究極のガイド：プレゼントをパーソナライズ  
*reference:* The ultimate guide to Secret Santa: Personalise the presents

| variant | hypothesis | sentence BLEU |
|---|---|---|
| dense | The ultimate guide to the secrets of the secret: a gift of the gift of the gift | 14.9 |
| moe_4e_top1 | The ultimate guide to the secret of the secret: the gift of personalized | 19.7 |
| moe_4e_top2 | The ultimate guide to the secret of the secret: personalizing the idea of the assets | 16.9 |
| moe_shared1_3e_top1 | The ultimate guide to the secret of the secret: Personalized for personalized | 21.4 |
| moe_4e_top2_active | The ultimate guide to the secret Sanタ: personalized for personalized | 26.0 |

**[ja→en, test #16098]** 6.小さい試験順序は歓迎されています。  
*reference:* 6. Small trial order is welcome.

| variant | hypothesis | sentence BLEU |
|---|---|---|
| dense | 6. Small trial order is welcome. | 100.0 |
| moe_4e_top1 | 6. Small trial order is welcome. | 100.0 |
| moe_4e_top2 | 6. Small trial order is welcome. | 100.0 |
| moe_shared1_3e_top1 | 6. Small trial order is welcome. | 100.0 |
| moe_4e_top2_active | 6. Small trial order is welcome. | 100.0 |

