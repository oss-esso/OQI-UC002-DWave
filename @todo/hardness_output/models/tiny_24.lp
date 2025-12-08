\ Model crop_optimization_tiny_24
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xf7b8ae2889585ab5
Maximize
  0.1887735 Y_Plot1_Wheat + 0.206367 Y_Plot1_Rice
   + 0.23656125 Y_Plot1_Soybeans + 0.24464475 Y_Plot1_Lentils
   + 0.22182075 Y_Plot1_Potatoes + 0.0661005 Y_Plot2_Wheat
   + 0.072261 Y_Plot2_Rice + 0.08283375 Y_Plot2_Soybeans
   + 0.08566425 Y_Plot2_Lentils + 0.07767225 Y_Plot2_Potatoes
   + 0.082973 Y_Plot3_Wheat + 0.090706 Y_Plot3_Rice
   + 0.1039775 Y_Plot3_Soybeans + 0.1075305 Y_Plot3_Lentils
   + 0.0974985 Y_Plot3_Potatoes + 0.059153 Y_Plot4_Wheat
   + 0.064666 Y_Plot4_Rice + 0.0741275 Y_Plot4_Soybeans
   + 0.0766605 Y_Plot4_Lentils + 0.0695085 Y_Plot4_Potatoes
Subject To
 Max_Assignment_Plot1: Y_Plot1_Wheat + Y_Plot1_Rice + Y_Plot1_Soybeans
   + Y_Plot1_Lentils + Y_Plot1_Potatoes <= 1
 Max_Assignment_Plot2: Y_Plot2_Wheat + Y_Plot2_Rice + Y_Plot2_Soybeans
   + Y_Plot2_Lentils + Y_Plot2_Potatoes <= 1
 Max_Assignment_Plot3: Y_Plot3_Wheat + Y_Plot3_Rice + Y_Plot3_Soybeans
   + Y_Plot3_Lentils + Y_Plot3_Potatoes <= 1
 Max_Assignment_Plot4: Y_Plot4_Wheat + Y_Plot4_Rice + Y_Plot4_Soybeans
   + Y_Plot4_Lentils + Y_Plot4_Potatoes <= 1
 U_Link_Plot1_Wheat: Y_Plot1_Wheat - U_Wheat <= 0
 U_Link_Plot2_Wheat: Y_Plot2_Wheat - U_Wheat <= 0
 U_Link_Plot3_Wheat: Y_Plot3_Wheat - U_Wheat <= 0
 U_Link_Plot4_Wheat: Y_Plot4_Wheat - U_Wheat <= 0
 U_Bound_Wheat: - Y_Plot1_Wheat - Y_Plot2_Wheat - Y_Plot3_Wheat
   - Y_Plot4_Wheat + U_Wheat <= 0
 U_Link_Plot1_Rice: Y_Plot1_Rice - U_Rice <= 0
 U_Link_Plot2_Rice: Y_Plot2_Rice - U_Rice <= 0
 U_Link_Plot3_Rice: Y_Plot3_Rice - U_Rice <= 0
 U_Link_Plot4_Rice: Y_Plot4_Rice - U_Rice <= 0
 U_Bound_Rice: - Y_Plot1_Rice - Y_Plot2_Rice - Y_Plot3_Rice - Y_Plot4_Rice
   + U_Rice <= 0
 U_Link_Plot1_Soybeans: Y_Plot1_Soybeans - U_Soybeans <= 0
 U_Link_Plot2_Soybeans: Y_Plot2_Soybeans - U_Soybeans <= 0
 U_Link_Plot3_Soybeans: Y_Plot3_Soybeans - U_Soybeans <= 0
 U_Link_Plot4_Soybeans: Y_Plot4_Soybeans - U_Soybeans <= 0
 U_Bound_Soybeans: - Y_Plot1_Soybeans - Y_Plot2_Soybeans - Y_Plot3_Soybeans
   - Y_Plot4_Soybeans + U_Soybeans <= 0
 U_Link_Plot1_Lentils: Y_Plot1_Lentils - U_Lentils <= 0
 U_Link_Plot2_Lentils: Y_Plot2_Lentils - U_Lentils <= 0
 U_Link_Plot3_Lentils: Y_Plot3_Lentils - U_Lentils <= 0
 U_Link_Plot4_Lentils: Y_Plot4_Lentils - U_Lentils <= 0
 U_Bound_Lentils: - Y_Plot1_Lentils - Y_Plot2_Lentils - Y_Plot3_Lentils
   - Y_Plot4_Lentils + U_Lentils <= 0
 U_Link_Plot1_Potatoes: Y_Plot1_Potatoes - U_Potatoes <= 0
 U_Link_Plot2_Potatoes: Y_Plot2_Potatoes - U_Potatoes <= 0
 U_Link_Plot3_Potatoes: Y_Plot3_Potatoes - U_Potatoes <= 0
 U_Link_Plot4_Potatoes: Y_Plot4_Potatoes - U_Potatoes <= 0
 U_Bound_Potatoes: - Y_Plot1_Potatoes - Y_Plot2_Potatoes - Y_Plot3_Potatoes
   - Y_Plot4_Potatoes + U_Potatoes <= 0
 MinFoodGroup_Grains: U_Wheat + U_Rice >= 1
 MinFoodGroup_Legumes: U_Soybeans + U_Lentils >= 1
 MinFoodGroup_Vegetables: U_Potatoes >= 1
Bounds
Binaries
 Y_Plot1_Wheat Y_Plot1_Rice Y_Plot1_Soybeans Y_Plot1_Lentils
 Y_Plot1_Potatoes Y_Plot2_Wheat Y_Plot2_Rice Y_Plot2_Soybeans
 Y_Plot2_Lentils Y_Plot2_Potatoes Y_Plot3_Wheat Y_Plot3_Rice
 Y_Plot3_Soybeans Y_Plot3_Lentils Y_Plot3_Potatoes Y_Plot4_Wheat
 Y_Plot4_Rice Y_Plot4_Soybeans Y_Plot4_Lentils Y_Plot4_Potatoes U_Wheat
 U_Rice U_Soybeans U_Lentils U_Potatoes
End
