\ Model crop_optimization_small_60
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x60b3e48553e1e9c3
Maximize
  0.0419615 Y_Plot1_Wheat + 0.0419089166666667 Y_Plot1_Rice
   + 0.0463785 Y_Plot1_Corn + 0.0548444166666667 Y_Plot1_Soybeans
   + 0.0526885 Y_Plot1_Lentils + 0.0474301666666667 Y_Plot1_Potatoes
   + 0.0419089166666667 Y_Plot1_Carrots + 0.04085725 Y_Plot1_Apples
   + 0.049609 Y_Plot2_Wheat + 0.0495468333333333 Y_Plot2_Rice
   + 0.054831 Y_Plot2_Corn + 0.0648398333333333 Y_Plot2_Soybeans
   + 0.062291 Y_Plot2_Lentils + 0.0560743333333333 Y_Plot2_Potatoes
   + 0.0495468333333333 Y_Plot2_Carrots + 0.0483035 Y_Plot2_Apples
   + 0.1191015 Y_Plot3_Wheat + 0.11895225 Y_Plot3_Rice
   + 0.1316385 Y_Plot3_Corn + 0.15566775 Y_Plot3_Soybeans
   + 0.1495485 Y_Plot3_Lentils + 0.1346235 Y_Plot3_Potatoes
   + 0.11895225 Y_Plot3_Carrots + 0.11596725 Y_Plot3_Apples
   + 0.035777 Y_Plot4_Wheat + 0.0357321666666667 Y_Plot4_Rice
   + 0.039543 Y_Plot4_Corn + 0.0467611666666667 Y_Plot4_Soybeans
   + 0.044923 Y_Plot4_Lentils + 0.0404396666666667 Y_Plot4_Potatoes
   + 0.0357321666666667 Y_Plot4_Carrots + 0.0348355 Y_Plot4_Apples
   + 0.0246715 Y_Plot5_Wheat + 0.0246405833333333 Y_Plot5_Rice
   + 0.0272685 Y_Plot5_Corn + 0.0322460833333333 Y_Plot5_Soybeans
   + 0.0309785 Y_Plot5_Lentils + 0.0278868333333333 Y_Plot5_Potatoes
   + 0.0246405833333333 Y_Plot5_Carrots + 0.02402225 Y_Plot5_Apples
   + 0.1278795 Y_Plot6_Wheat + 0.12771925 Y_Plot6_Rice
   + 0.1413405 Y_Plot6_Corn + 0.16714075 Y_Plot6_Soybeans
   + 0.1605705 Y_Plot6_Lentils + 0.1445455 Y_Plot6_Potatoes
   + 0.12771925 Y_Plot6_Carrots + 0.12451425 Y_Plot6_Apples
Subject To
 Max_Assignment_Plot1: Y_Plot1_Wheat + Y_Plot1_Rice + Y_Plot1_Corn
   + Y_Plot1_Soybeans + Y_Plot1_Lentils + Y_Plot1_Potatoes
   + Y_Plot1_Carrots + Y_Plot1_Apples <= 1
 Max_Assignment_Plot2: Y_Plot2_Wheat + Y_Plot2_Rice + Y_Plot2_Corn
   + Y_Plot2_Soybeans + Y_Plot2_Lentils + Y_Plot2_Potatoes
   + Y_Plot2_Carrots + Y_Plot2_Apples <= 1
 Max_Assignment_Plot3: Y_Plot3_Wheat + Y_Plot3_Rice + Y_Plot3_Corn
   + Y_Plot3_Soybeans + Y_Plot3_Lentils + Y_Plot3_Potatoes
   + Y_Plot3_Carrots + Y_Plot3_Apples <= 1
 Max_Assignment_Plot4: Y_Plot4_Wheat + Y_Plot4_Rice + Y_Plot4_Corn
   + Y_Plot4_Soybeans + Y_Plot4_Lentils + Y_Plot4_Potatoes
   + Y_Plot4_Carrots + Y_Plot4_Apples <= 1
 Max_Assignment_Plot5: Y_Plot5_Wheat + Y_Plot5_Rice + Y_Plot5_Corn
   + Y_Plot5_Soybeans + Y_Plot5_Lentils + Y_Plot5_Potatoes
   + Y_Plot5_Carrots + Y_Plot5_Apples <= 1
 Max_Assignment_Plot6: Y_Plot6_Wheat + Y_Plot6_Rice + Y_Plot6_Corn
   + Y_Plot6_Soybeans + Y_Plot6_Lentils + Y_Plot6_Potatoes
   + Y_Plot6_Carrots + Y_Plot6_Apples <= 1
 U_Link_Plot1_Wheat: Y_Plot1_Wheat - U_Wheat <= 0
 U_Link_Plot2_Wheat: Y_Plot2_Wheat - U_Wheat <= 0
 U_Link_Plot3_Wheat: Y_Plot3_Wheat - U_Wheat <= 0
 U_Link_Plot4_Wheat: Y_Plot4_Wheat - U_Wheat <= 0
 U_Link_Plot5_Wheat: Y_Plot5_Wheat - U_Wheat <= 0
 U_Link_Plot6_Wheat: Y_Plot6_Wheat - U_Wheat <= 0
 U_Bound_Wheat: - Y_Plot1_Wheat - Y_Plot2_Wheat - Y_Plot3_Wheat
   - Y_Plot4_Wheat - Y_Plot5_Wheat - Y_Plot6_Wheat + U_Wheat <= 0
 U_Link_Plot1_Rice: Y_Plot1_Rice - U_Rice <= 0
 U_Link_Plot2_Rice: Y_Plot2_Rice - U_Rice <= 0
 U_Link_Plot3_Rice: Y_Plot3_Rice - U_Rice <= 0
 U_Link_Plot4_Rice: Y_Plot4_Rice - U_Rice <= 0
 U_Link_Plot5_Rice: Y_Plot5_Rice - U_Rice <= 0
 U_Link_Plot6_Rice: Y_Plot6_Rice - U_Rice <= 0
 U_Bound_Rice: - Y_Plot1_Rice - Y_Plot2_Rice - Y_Plot3_Rice - Y_Plot4_Rice
   - Y_Plot5_Rice - Y_Plot6_Rice + U_Rice <= 0
 U_Link_Plot1_Corn: Y_Plot1_Corn - U_Corn <= 0
 U_Link_Plot2_Corn: Y_Plot2_Corn - U_Corn <= 0
 U_Link_Plot3_Corn: Y_Plot3_Corn - U_Corn <= 0
 U_Link_Plot4_Corn: Y_Plot4_Corn - U_Corn <= 0
 U_Link_Plot5_Corn: Y_Plot5_Corn - U_Corn <= 0
 U_Link_Plot6_Corn: Y_Plot6_Corn - U_Corn <= 0
 U_Bound_Corn: - Y_Plot1_Corn - Y_Plot2_Corn - Y_Plot3_Corn - Y_Plot4_Corn
   - Y_Plot5_Corn - Y_Plot6_Corn + U_Corn <= 0
 U_Link_Plot1_Soybeans: Y_Plot1_Soybeans - U_Soybeans <= 0
 U_Link_Plot2_Soybeans: Y_Plot2_Soybeans - U_Soybeans <= 0
 U_Link_Plot3_Soybeans: Y_Plot3_Soybeans - U_Soybeans <= 0
 U_Link_Plot4_Soybeans: Y_Plot4_Soybeans - U_Soybeans <= 0
 U_Link_Plot5_Soybeans: Y_Plot5_Soybeans - U_Soybeans <= 0
 U_Link_Plot6_Soybeans: Y_Plot6_Soybeans - U_Soybeans <= 0
 U_Bound_Soybeans: - Y_Plot1_Soybeans - Y_Plot2_Soybeans - Y_Plot3_Soybeans
   - Y_Plot4_Soybeans - Y_Plot5_Soybeans - Y_Plot6_Soybeans + U_Soybeans
   <= 0
 U_Link_Plot1_Lentils: Y_Plot1_Lentils - U_Lentils <= 0
 U_Link_Plot2_Lentils: Y_Plot2_Lentils - U_Lentils <= 0
 U_Link_Plot3_Lentils: Y_Plot3_Lentils - U_Lentils <= 0
 U_Link_Plot4_Lentils: Y_Plot4_Lentils - U_Lentils <= 0
 U_Link_Plot5_Lentils: Y_Plot5_Lentils - U_Lentils <= 0
 U_Link_Plot6_Lentils: Y_Plot6_Lentils - U_Lentils <= 0
 U_Bound_Lentils: - Y_Plot1_Lentils - Y_Plot2_Lentils - Y_Plot3_Lentils
   - Y_Plot4_Lentils - Y_Plot5_Lentils - Y_Plot6_Lentils + U_Lentils <= 0
 U_Link_Plot1_Potatoes: Y_Plot1_Potatoes - U_Potatoes <= 0
 U_Link_Plot2_Potatoes: Y_Plot2_Potatoes - U_Potatoes <= 0
 U_Link_Plot3_Potatoes: Y_Plot3_Potatoes - U_Potatoes <= 0
 U_Link_Plot4_Potatoes: Y_Plot4_Potatoes - U_Potatoes <= 0
 U_Link_Plot5_Potatoes: Y_Plot5_Potatoes - U_Potatoes <= 0
 U_Link_Plot6_Potatoes: Y_Plot6_Potatoes - U_Potatoes <= 0
 U_Bound_Potatoes: - Y_Plot1_Potatoes - Y_Plot2_Potatoes - Y_Plot3_Potatoes
   - Y_Plot4_Potatoes - Y_Plot5_Potatoes - Y_Plot6_Potatoes + U_Potatoes
   <= 0
 U_Link_Plot1_Carrots: Y_Plot1_Carrots - U_Carrots <= 0
 U_Link_Plot2_Carrots: Y_Plot2_Carrots - U_Carrots <= 0
 U_Link_Plot3_Carrots: Y_Plot3_Carrots - U_Carrots <= 0
 U_Link_Plot4_Carrots: Y_Plot4_Carrots - U_Carrots <= 0
 U_Link_Plot5_Carrots: Y_Plot5_Carrots - U_Carrots <= 0
 U_Link_Plot6_Carrots: Y_Plot6_Carrots - U_Carrots <= 0
 U_Bound_Carrots: - Y_Plot1_Carrots - Y_Plot2_Carrots - Y_Plot3_Carrots
   - Y_Plot4_Carrots - Y_Plot5_Carrots - Y_Plot6_Carrots + U_Carrots <= 0
 U_Link_Plot1_Apples: Y_Plot1_Apples - U_Apples <= 0
 U_Link_Plot2_Apples: Y_Plot2_Apples - U_Apples <= 0
 U_Link_Plot3_Apples: Y_Plot3_Apples - U_Apples <= 0
 U_Link_Plot4_Apples: Y_Plot4_Apples - U_Apples <= 0
 U_Link_Plot5_Apples: Y_Plot5_Apples - U_Apples <= 0
 U_Link_Plot6_Apples: Y_Plot6_Apples - U_Apples <= 0
 U_Bound_Apples: - Y_Plot1_Apples - Y_Plot2_Apples - Y_Plot3_Apples
   - Y_Plot4_Apples - Y_Plot5_Apples - Y_Plot6_Apples + U_Apples <= 0
 MinFoodGroup_Grains: U_Wheat + U_Rice + U_Corn >= 1
 MinFoodGroup_Legumes: U_Soybeans + U_Lentils >= 1
 MinFoodGroup_Vegetables: U_Potatoes + U_Carrots >= 1
 MinFoodGroup_Fruits: U_Apples >= 1
Bounds
Binaries
 Y_Plot1_Wheat Y_Plot1_Rice Y_Plot1_Corn Y_Plot1_Soybeans Y_Plot1_Lentils
 Y_Plot1_Potatoes Y_Plot1_Carrots Y_Plot1_Apples Y_Plot2_Wheat Y_Plot2_Rice
 Y_Plot2_Corn Y_Plot2_Soybeans Y_Plot2_Lentils Y_Plot2_Potatoes
 Y_Plot2_Carrots Y_Plot2_Apples Y_Plot3_Wheat Y_Plot3_Rice Y_Plot3_Corn
 Y_Plot3_Soybeans Y_Plot3_Lentils Y_Plot3_Potatoes Y_Plot3_Carrots
 Y_Plot3_Apples Y_Plot4_Wheat Y_Plot4_Rice Y_Plot4_Corn Y_Plot4_Soybeans
 Y_Plot4_Lentils Y_Plot4_Potatoes Y_Plot4_Carrots Y_Plot4_Apples
 Y_Plot5_Wheat Y_Plot5_Rice Y_Plot5_Corn Y_Plot5_Soybeans Y_Plot5_Lentils
 Y_Plot5_Potatoes Y_Plot5_Carrots Y_Plot5_Apples Y_Plot6_Wheat Y_Plot6_Rice
 Y_Plot6_Corn Y_Plot6_Soybeans Y_Plot6_Lentils Y_Plot6_Potatoes
 Y_Plot6_Carrots Y_Plot6_Apples U_Wheat U_Rice U_Corn U_Soybeans U_Lentils
 U_Potatoes U_Carrots U_Apples
End
