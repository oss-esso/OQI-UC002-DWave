\ Model crop_optimization_15plots
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xa9db16cfe4049494
Maximize
  0.0296666666666667 Y_Patch1_Wheat + 0.025 Y_Patch1_Corn
   + 0.0246666666666667 Y_Patch1_Rice
   + 0.0363333333333333 Y_Patch1_Soybeans
   + 0.0246666666666667 Y_Patch1_Potatoes
   + 0.0293333333333333 Y_Patch1_Apples + 0.0296666666666667 Y_Patch2_Wheat
   + 0.025 Y_Patch2_Corn + 0.0246666666666667 Y_Patch2_Rice
   + 0.0363333333333333 Y_Patch2_Soybeans
   + 0.0246666666666667 Y_Patch2_Potatoes
   + 0.0293333333333333 Y_Patch2_Apples + 0.0296666666666667 Y_Patch3_Wheat
   + 0.025 Y_Patch3_Corn + 0.0246666666666667 Y_Patch3_Rice
   + 0.0363333333333333 Y_Patch3_Soybeans
   + 0.0246666666666667 Y_Patch3_Potatoes
   + 0.0293333333333333 Y_Patch3_Apples + 0.0296666666666667 Y_Patch4_Wheat
   + 0.025 Y_Patch4_Corn + 0.0246666666666667 Y_Patch4_Rice
   + 0.0363333333333333 Y_Patch4_Soybeans
   + 0.0246666666666667 Y_Patch4_Potatoes
   + 0.0293333333333333 Y_Patch4_Apples + 0.0296666666666667 Y_Patch5_Wheat
   + 0.025 Y_Patch5_Corn + 0.0246666666666667 Y_Patch5_Rice
   + 0.0363333333333333 Y_Patch5_Soybeans
   + 0.0246666666666667 Y_Patch5_Potatoes
   + 0.0293333333333333 Y_Patch5_Apples + 0.0296666666666667 Y_Patch6_Wheat
   + 0.025 Y_Patch6_Corn + 0.0246666666666667 Y_Patch6_Rice
   + 0.0363333333333333 Y_Patch6_Soybeans
   + 0.0246666666666667 Y_Patch6_Potatoes
   + 0.0293333333333333 Y_Patch6_Apples + 0.0296666666666667 Y_Patch7_Wheat
   + 0.025 Y_Patch7_Corn + 0.0246666666666667 Y_Patch7_Rice
   + 0.0363333333333333 Y_Patch7_Soybeans
   + 0.0246666666666667 Y_Patch7_Potatoes
   + 0.0293333333333333 Y_Patch7_Apples + 0.0296666666666667 Y_Patch8_Wheat
   + 0.025 Y_Patch8_Corn + 0.0246666666666667 Y_Patch8_Rice
   + 0.0363333333333333 Y_Patch8_Soybeans
   + 0.0246666666666667 Y_Patch8_Potatoes
   + 0.0293333333333333 Y_Patch8_Apples + 0.0296666666666667 Y_Patch9_Wheat
   + 0.025 Y_Patch9_Corn + 0.0246666666666667 Y_Patch9_Rice
   + 0.0363333333333333 Y_Patch9_Soybeans
   + 0.0246666666666667 Y_Patch9_Potatoes
   + 0.0293333333333333 Y_Patch9_Apples
   + 0.0296666666666667 Y_Patch10_Wheat + 0.025 Y_Patch10_Corn
   + 0.0246666666666667 Y_Patch10_Rice
   + 0.0363333333333333 Y_Patch10_Soybeans
   + 0.0246666666666667 Y_Patch10_Potatoes
   + 0.0293333333333333 Y_Patch10_Apples
   + 0.0296666666666667 Y_Patch11_Wheat + 0.025 Y_Patch11_Corn
   + 0.0246666666666667 Y_Patch11_Rice
   + 0.0363333333333333 Y_Patch11_Soybeans
   + 0.0246666666666667 Y_Patch11_Potatoes
   + 0.0293333333333333 Y_Patch11_Apples
   + 0.0296666666666667 Y_Patch12_Wheat + 0.025 Y_Patch12_Corn
   + 0.0246666666666667 Y_Patch12_Rice
   + 0.0363333333333333 Y_Patch12_Soybeans
   + 0.0246666666666667 Y_Patch12_Potatoes
   + 0.0293333333333333 Y_Patch12_Apples
   + 0.0296666666666667 Y_Patch13_Wheat + 0.025 Y_Patch13_Corn
   + 0.0246666666666667 Y_Patch13_Rice
   + 0.0363333333333333 Y_Patch13_Soybeans
   + 0.0246666666666667 Y_Patch13_Potatoes
   + 0.0293333333333333 Y_Patch13_Apples
   + 0.0296666666666667 Y_Patch14_Wheat + 0.025 Y_Patch14_Corn
   + 0.0246666666666667 Y_Patch14_Rice
   + 0.0363333333333333 Y_Patch14_Soybeans
   + 0.0246666666666667 Y_Patch14_Potatoes
   + 0.0293333333333333 Y_Patch14_Apples
   + 0.0296666666666667 Y_Patch15_Wheat + 0.025 Y_Patch15_Corn
   + 0.0246666666666667 Y_Patch15_Rice
   + 0.0363333333333333 Y_Patch15_Soybeans
   + 0.0246666666666667 Y_Patch15_Potatoes
   + 0.0293333333333333 Y_Patch15_Apples
Subject To
 Max_Assignment_Patch1: Y_Patch1_Wheat + Y_Patch1_Corn + Y_Patch1_Rice
   + Y_Patch1_Soybeans + Y_Patch1_Potatoes + Y_Patch1_Apples <= 1
 Max_Assignment_Patch2: Y_Patch2_Wheat + Y_Patch2_Corn + Y_Patch2_Rice
   + Y_Patch2_Soybeans + Y_Patch2_Potatoes + Y_Patch2_Apples <= 1
 Max_Assignment_Patch3: Y_Patch3_Wheat + Y_Patch3_Corn + Y_Patch3_Rice
   + Y_Patch3_Soybeans + Y_Patch3_Potatoes + Y_Patch3_Apples <= 1
 Max_Assignment_Patch4: Y_Patch4_Wheat + Y_Patch4_Corn + Y_Patch4_Rice
   + Y_Patch4_Soybeans + Y_Patch4_Potatoes + Y_Patch4_Apples <= 1
 Max_Assignment_Patch5: Y_Patch5_Wheat + Y_Patch5_Corn + Y_Patch5_Rice
   + Y_Patch5_Soybeans + Y_Patch5_Potatoes + Y_Patch5_Apples <= 1
 Max_Assignment_Patch6: Y_Patch6_Wheat + Y_Patch6_Corn + Y_Patch6_Rice
   + Y_Patch6_Soybeans + Y_Patch6_Potatoes + Y_Patch6_Apples <= 1
 Max_Assignment_Patch7: Y_Patch7_Wheat + Y_Patch7_Corn + Y_Patch7_Rice
   + Y_Patch7_Soybeans + Y_Patch7_Potatoes + Y_Patch7_Apples <= 1
 Max_Assignment_Patch8: Y_Patch8_Wheat + Y_Patch8_Corn + Y_Patch8_Rice
   + Y_Patch8_Soybeans + Y_Patch8_Potatoes + Y_Patch8_Apples <= 1
 Max_Assignment_Patch9: Y_Patch9_Wheat + Y_Patch9_Corn + Y_Patch9_Rice
   + Y_Patch9_Soybeans + Y_Patch9_Potatoes + Y_Patch9_Apples <= 1
 Max_Assignment_Patch10: Y_Patch10_Wheat + Y_Patch10_Corn + Y_Patch10_Rice
   + Y_Patch10_Soybeans + Y_Patch10_Potatoes + Y_Patch10_Apples <= 1
 Max_Assignment_Patch11: Y_Patch11_Wheat + Y_Patch11_Corn + Y_Patch11_Rice
   + Y_Patch11_Soybeans + Y_Patch11_Potatoes + Y_Patch11_Apples <= 1
 Max_Assignment_Patch12: Y_Patch12_Wheat + Y_Patch12_Corn + Y_Patch12_Rice
   + Y_Patch12_Soybeans + Y_Patch12_Potatoes + Y_Patch12_Apples <= 1
 Max_Assignment_Patch13: Y_Patch13_Wheat + Y_Patch13_Corn + Y_Patch13_Rice
   + Y_Patch13_Soybeans + Y_Patch13_Potatoes + Y_Patch13_Apples <= 1
 Max_Assignment_Patch14: Y_Patch14_Wheat + Y_Patch14_Corn + Y_Patch14_Rice
   + Y_Patch14_Soybeans + Y_Patch14_Potatoes + Y_Patch14_Apples <= 1
 Max_Assignment_Patch15: Y_Patch15_Wheat + Y_Patch15_Corn + Y_Patch15_Rice
   + Y_Patch15_Soybeans + Y_Patch15_Potatoes + Y_Patch15_Apples <= 1
 U_Link_Patch1_Wheat: Y_Patch1_Wheat - U_Wheat <= 0
 U_Link_Patch2_Wheat: Y_Patch2_Wheat - U_Wheat <= 0
 U_Link_Patch3_Wheat: Y_Patch3_Wheat - U_Wheat <= 0
 U_Link_Patch4_Wheat: Y_Patch4_Wheat - U_Wheat <= 0
 U_Link_Patch5_Wheat: Y_Patch5_Wheat - U_Wheat <= 0
 U_Link_Patch6_Wheat: Y_Patch6_Wheat - U_Wheat <= 0
 U_Link_Patch7_Wheat: Y_Patch7_Wheat - U_Wheat <= 0
 U_Link_Patch8_Wheat: Y_Patch8_Wheat - U_Wheat <= 0
 U_Link_Patch9_Wheat: Y_Patch9_Wheat - U_Wheat <= 0
 U_Link_Patch10_Wheat: Y_Patch10_Wheat - U_Wheat <= 0
 U_Link_Patch11_Wheat: Y_Patch11_Wheat - U_Wheat <= 0
 U_Link_Patch12_Wheat: Y_Patch12_Wheat - U_Wheat <= 0
 U_Link_Patch13_Wheat: Y_Patch13_Wheat - U_Wheat <= 0
 U_Link_Patch14_Wheat: Y_Patch14_Wheat - U_Wheat <= 0
 U_Link_Patch15_Wheat: Y_Patch15_Wheat - U_Wheat <= 0
 U_Bound_Wheat: - Y_Patch1_Wheat - Y_Patch2_Wheat - Y_Patch3_Wheat
   - Y_Patch4_Wheat - Y_Patch5_Wheat - Y_Patch6_Wheat - Y_Patch7_Wheat
   - Y_Patch8_Wheat - Y_Patch9_Wheat - Y_Patch10_Wheat - Y_Patch11_Wheat
   - Y_Patch12_Wheat - Y_Patch13_Wheat - Y_Patch14_Wheat - Y_Patch15_Wheat
   + U_Wheat <= 0
 U_Link_Patch1_Corn: Y_Patch1_Corn - U_Corn <= 0
 U_Link_Patch2_Corn: Y_Patch2_Corn - U_Corn <= 0
 U_Link_Patch3_Corn: Y_Patch3_Corn - U_Corn <= 0
 U_Link_Patch4_Corn: Y_Patch4_Corn - U_Corn <= 0
 U_Link_Patch5_Corn: Y_Patch5_Corn - U_Corn <= 0
 U_Link_Patch6_Corn: Y_Patch6_Corn - U_Corn <= 0
 U_Link_Patch7_Corn: Y_Patch7_Corn - U_Corn <= 0
 U_Link_Patch8_Corn: Y_Patch8_Corn - U_Corn <= 0
 U_Link_Patch9_Corn: Y_Patch9_Corn - U_Corn <= 0
 U_Link_Patch10_Corn: Y_Patch10_Corn - U_Corn <= 0
 U_Link_Patch11_Corn: Y_Patch11_Corn - U_Corn <= 0
 U_Link_Patch12_Corn: Y_Patch12_Corn - U_Corn <= 0
 U_Link_Patch13_Corn: Y_Patch13_Corn - U_Corn <= 0
 U_Link_Patch14_Corn: Y_Patch14_Corn - U_Corn <= 0
 U_Link_Patch15_Corn: Y_Patch15_Corn - U_Corn <= 0
 U_Bound_Corn: - Y_Patch1_Corn - Y_Patch2_Corn - Y_Patch3_Corn
   - Y_Patch4_Corn - Y_Patch5_Corn - Y_Patch6_Corn - Y_Patch7_Corn
   - Y_Patch8_Corn - Y_Patch9_Corn - Y_Patch10_Corn - Y_Patch11_Corn
   - Y_Patch12_Corn - Y_Patch13_Corn - Y_Patch14_Corn - Y_Patch15_Corn
   + U_Corn <= 0
 U_Link_Patch1_Rice: Y_Patch1_Rice - U_Rice <= 0
 U_Link_Patch2_Rice: Y_Patch2_Rice - U_Rice <= 0
 U_Link_Patch3_Rice: Y_Patch3_Rice - U_Rice <= 0
 U_Link_Patch4_Rice: Y_Patch4_Rice - U_Rice <= 0
 U_Link_Patch5_Rice: Y_Patch5_Rice - U_Rice <= 0
 U_Link_Patch6_Rice: Y_Patch6_Rice - U_Rice <= 0
 U_Link_Patch7_Rice: Y_Patch7_Rice - U_Rice <= 0
 U_Link_Patch8_Rice: Y_Patch8_Rice - U_Rice <= 0
 U_Link_Patch9_Rice: Y_Patch9_Rice - U_Rice <= 0
 U_Link_Patch10_Rice: Y_Patch10_Rice - U_Rice <= 0
 U_Link_Patch11_Rice: Y_Patch11_Rice - U_Rice <= 0
 U_Link_Patch12_Rice: Y_Patch12_Rice - U_Rice <= 0
 U_Link_Patch13_Rice: Y_Patch13_Rice - U_Rice <= 0
 U_Link_Patch14_Rice: Y_Patch14_Rice - U_Rice <= 0
 U_Link_Patch15_Rice: Y_Patch15_Rice - U_Rice <= 0
 U_Bound_Rice: - Y_Patch1_Rice - Y_Patch2_Rice - Y_Patch3_Rice
   - Y_Patch4_Rice - Y_Patch5_Rice - Y_Patch6_Rice - Y_Patch7_Rice
   - Y_Patch8_Rice - Y_Patch9_Rice - Y_Patch10_Rice - Y_Patch11_Rice
   - Y_Patch12_Rice - Y_Patch13_Rice - Y_Patch14_Rice - Y_Patch15_Rice
   + U_Rice <= 0
 U_Link_Patch1_Soybeans: Y_Patch1_Soybeans - U_Soybeans <= 0
 U_Link_Patch2_Soybeans: Y_Patch2_Soybeans - U_Soybeans <= 0
 U_Link_Patch3_Soybeans: Y_Patch3_Soybeans - U_Soybeans <= 0
 U_Link_Patch4_Soybeans: Y_Patch4_Soybeans - U_Soybeans <= 0
 U_Link_Patch5_Soybeans: Y_Patch5_Soybeans - U_Soybeans <= 0
 U_Link_Patch6_Soybeans: Y_Patch6_Soybeans - U_Soybeans <= 0
 U_Link_Patch7_Soybeans: Y_Patch7_Soybeans - U_Soybeans <= 0
 U_Link_Patch8_Soybeans: Y_Patch8_Soybeans - U_Soybeans <= 0
 U_Link_Patch9_Soybeans: Y_Patch9_Soybeans - U_Soybeans <= 0
 U_Link_Patch10_Soybeans: Y_Patch10_Soybeans - U_Soybeans <= 0
 U_Link_Patch11_Soybeans: Y_Patch11_Soybeans - U_Soybeans <= 0
 U_Link_Patch12_Soybeans: Y_Patch12_Soybeans - U_Soybeans <= 0
 U_Link_Patch13_Soybeans: Y_Patch13_Soybeans - U_Soybeans <= 0
 U_Link_Patch14_Soybeans: Y_Patch14_Soybeans - U_Soybeans <= 0
 U_Link_Patch15_Soybeans: Y_Patch15_Soybeans - U_Soybeans <= 0
 U_Bound_Soybeans: - Y_Patch1_Soybeans - Y_Patch2_Soybeans
   - Y_Patch3_Soybeans - Y_Patch4_Soybeans - Y_Patch5_Soybeans
   - Y_Patch6_Soybeans - Y_Patch7_Soybeans - Y_Patch8_Soybeans
   - Y_Patch9_Soybeans - Y_Patch10_Soybeans - Y_Patch11_Soybeans
   - Y_Patch12_Soybeans - Y_Patch13_Soybeans - Y_Patch14_Soybeans
   - Y_Patch15_Soybeans + U_Soybeans <= 0
 U_Link_Patch1_Potatoes: Y_Patch1_Potatoes - U_Potatoes <= 0
 U_Link_Patch2_Potatoes: Y_Patch2_Potatoes - U_Potatoes <= 0
 U_Link_Patch3_Potatoes: Y_Patch3_Potatoes - U_Potatoes <= 0
 U_Link_Patch4_Potatoes: Y_Patch4_Potatoes - U_Potatoes <= 0
 U_Link_Patch5_Potatoes: Y_Patch5_Potatoes - U_Potatoes <= 0
 U_Link_Patch6_Potatoes: Y_Patch6_Potatoes - U_Potatoes <= 0
 U_Link_Patch7_Potatoes: Y_Patch7_Potatoes - U_Potatoes <= 0
 U_Link_Patch8_Potatoes: Y_Patch8_Potatoes - U_Potatoes <= 0
 U_Link_Patch9_Potatoes: Y_Patch9_Potatoes - U_Potatoes <= 0
 U_Link_Patch10_Potatoes: Y_Patch10_Potatoes - U_Potatoes <= 0
 U_Link_Patch11_Potatoes: Y_Patch11_Potatoes - U_Potatoes <= 0
 U_Link_Patch12_Potatoes: Y_Patch12_Potatoes - U_Potatoes <= 0
 U_Link_Patch13_Potatoes: Y_Patch13_Potatoes - U_Potatoes <= 0
 U_Link_Patch14_Potatoes: Y_Patch14_Potatoes - U_Potatoes <= 0
 U_Link_Patch15_Potatoes: Y_Patch15_Potatoes - U_Potatoes <= 0
 U_Bound_Potatoes: - Y_Patch1_Potatoes - Y_Patch2_Potatoes
   - Y_Patch3_Potatoes - Y_Patch4_Potatoes - Y_Patch5_Potatoes
   - Y_Patch6_Potatoes - Y_Patch7_Potatoes - Y_Patch8_Potatoes
   - Y_Patch9_Potatoes - Y_Patch10_Potatoes - Y_Patch11_Potatoes
   - Y_Patch12_Potatoes - Y_Patch13_Potatoes - Y_Patch14_Potatoes
   - Y_Patch15_Potatoes + U_Potatoes <= 0
 U_Link_Patch1_Apples: Y_Patch1_Apples - U_Apples <= 0
 U_Link_Patch2_Apples: Y_Patch2_Apples - U_Apples <= 0
 U_Link_Patch3_Apples: Y_Patch3_Apples - U_Apples <= 0
 U_Link_Patch4_Apples: Y_Patch4_Apples - U_Apples <= 0
 U_Link_Patch5_Apples: Y_Patch5_Apples - U_Apples <= 0
 U_Link_Patch6_Apples: Y_Patch6_Apples - U_Apples <= 0
 U_Link_Patch7_Apples: Y_Patch7_Apples - U_Apples <= 0
 U_Link_Patch8_Apples: Y_Patch8_Apples - U_Apples <= 0
 U_Link_Patch9_Apples: Y_Patch9_Apples - U_Apples <= 0
 U_Link_Patch10_Apples: Y_Patch10_Apples - U_Apples <= 0
 U_Link_Patch11_Apples: Y_Patch11_Apples - U_Apples <= 0
 U_Link_Patch12_Apples: Y_Patch12_Apples - U_Apples <= 0
 U_Link_Patch13_Apples: Y_Patch13_Apples - U_Apples <= 0
 U_Link_Patch14_Apples: Y_Patch14_Apples - U_Apples <= 0
 U_Link_Patch15_Apples: Y_Patch15_Apples - U_Apples <= 0
 U_Bound_Apples: - Y_Patch1_Apples - Y_Patch2_Apples - Y_Patch3_Apples
   - Y_Patch4_Apples - Y_Patch5_Apples - Y_Patch6_Apples - Y_Patch7_Apples
   - Y_Patch8_Apples - Y_Patch9_Apples - Y_Patch10_Apples
   - Y_Patch11_Apples - Y_Patch12_Apples - Y_Patch13_Apples
   - Y_Patch14_Apples - Y_Patch15_Apples + U_Apples <= 0
 MinFoodGroup_Grains: U_Wheat + U_Corn + U_Rice >= 1
 MinFoodGroup_Legumes: U_Soybeans >= 1
 MinFoodGroup_Vegetables: U_Potatoes >= 1
 MinFoodGroup_Fruits: U_Apples >= 1
Bounds
Binaries
 Y_Patch1_Wheat Y_Patch1_Corn Y_Patch1_Rice Y_Patch1_Soybeans
 Y_Patch1_Potatoes Y_Patch1_Apples Y_Patch2_Wheat Y_Patch2_Corn
 Y_Patch2_Rice Y_Patch2_Soybeans Y_Patch2_Potatoes Y_Patch2_Apples
 Y_Patch3_Wheat Y_Patch3_Corn Y_Patch3_Rice Y_Patch3_Soybeans
 Y_Patch3_Potatoes Y_Patch3_Apples Y_Patch4_Wheat Y_Patch4_Corn
 Y_Patch4_Rice Y_Patch4_Soybeans Y_Patch4_Potatoes Y_Patch4_Apples
 Y_Patch5_Wheat Y_Patch5_Corn Y_Patch5_Rice Y_Patch5_Soybeans
 Y_Patch5_Potatoes Y_Patch5_Apples Y_Patch6_Wheat Y_Patch6_Corn
 Y_Patch6_Rice Y_Patch6_Soybeans Y_Patch6_Potatoes Y_Patch6_Apples
 Y_Patch7_Wheat Y_Patch7_Corn Y_Patch7_Rice Y_Patch7_Soybeans
 Y_Patch7_Potatoes Y_Patch7_Apples Y_Patch8_Wheat Y_Patch8_Corn
 Y_Patch8_Rice Y_Patch8_Soybeans Y_Patch8_Potatoes Y_Patch8_Apples
 Y_Patch9_Wheat Y_Patch9_Corn Y_Patch9_Rice Y_Patch9_Soybeans
 Y_Patch9_Potatoes Y_Patch9_Apples Y_Patch10_Wheat Y_Patch10_Corn
 Y_Patch10_Rice Y_Patch10_Soybeans Y_Patch10_Potatoes Y_Patch10_Apples
 Y_Patch11_Wheat Y_Patch11_Corn Y_Patch11_Rice Y_Patch11_Soybeans
 Y_Patch11_Potatoes Y_Patch11_Apples Y_Patch12_Wheat Y_Patch12_Corn
 Y_Patch12_Rice Y_Patch12_Soybeans Y_Patch12_Potatoes Y_Patch12_Apples
 Y_Patch13_Wheat Y_Patch13_Corn Y_Patch13_Rice Y_Patch13_Soybeans
 Y_Patch13_Potatoes Y_Patch13_Apples Y_Patch14_Wheat Y_Patch14_Corn
 Y_Patch14_Rice Y_Patch14_Soybeans Y_Patch14_Potatoes Y_Patch14_Apples
 Y_Patch15_Wheat Y_Patch15_Corn Y_Patch15_Rice Y_Patch15_Soybeans
 Y_Patch15_Potatoes Y_Patch15_Apples U_Wheat U_Corn U_Rice U_Soybeans
 U_Potatoes U_Apples
End
