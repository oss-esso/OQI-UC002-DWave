\ Model crop_optimization_micro_6
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x4cc99ecd45b22392
Maximize
  0.1411805 Y_Plot1_Wheat + 0.1549815 Y_Plot1_Rice + 0.2373195 Y_Plot2_Wheat
   + 0.2605185 Y_Plot2_Rice
Subject To
 Max_Assignment_Plot1: Y_Plot1_Wheat + Y_Plot1_Rice <= 1
 Max_Assignment_Plot2: Y_Plot2_Wheat + Y_Plot2_Rice <= 1
 U_Link_Plot1_Wheat: Y_Plot1_Wheat - U_Wheat <= 0
 U_Link_Plot2_Wheat: Y_Plot2_Wheat - U_Wheat <= 0
 U_Bound_Wheat: - Y_Plot1_Wheat - Y_Plot2_Wheat + U_Wheat <= 0
 U_Link_Plot1_Rice: Y_Plot1_Rice - U_Rice <= 0
 U_Link_Plot2_Rice: Y_Plot2_Rice - U_Rice <= 0
 U_Bound_Rice: - Y_Plot1_Rice - Y_Plot2_Rice + U_Rice <= 0
 MinFoodGroup_Grains: U_Wheat + U_Rice >= 1
Bounds
Binaries
 Y_Plot1_Wheat Y_Plot1_Rice Y_Plot2_Wheat Y_Plot2_Rice U_Wheat U_Rice
End
