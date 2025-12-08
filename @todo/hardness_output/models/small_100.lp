\ Model crop_optimization_small_100
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x977337aaa7a07bba
Maximize
  0.0187476565429321 Y_Plot1_Wheat + 0.0171541057367829 Y_Plot1_Rice
   + 0.0190757405324334 Y_Plot1_Corn + 0.026527934008249 Y_Plot1_Soybeans
   + 0.0232939632545932 Y_Plot1_Lentils
   + 0.0192632170978628 Y_Plot1_Potatoes
   + 0.021278590176228 Y_Plot1_Carrots
   + 0.0253796400449944 Y_Plot1_Tomatoes
   + 0.0188882639670041 Y_Plot1_Apples + 0.0210911136107986 Y_Plot1_Oranges
   + 0.0195444319460067 Y_Plot1_Eggs + 0.0533933258342707 Y_Plot2_Wheat
   + 0.0488548931383577 Y_Plot2_Rice + 0.0543277090363705 Y_Plot2_Corn
   + 0.0755515560554931 Y_Plot2_Soybeans
   + 0.0663412073490814 Y_Plot2_Lentils
   + 0.0548616422947131 Y_Plot2_Potatoes
   + 0.0606014248218973 Y_Plot2_Carrots
   + 0.072281214848144 Y_Plot2_Tomatoes + 0.0537937757780277 Y_Plot2_Apples
   + 0.0600674915635546 Y_Plot2_Oranges + 0.0556625421822272 Y_Plot2_Eggs
   + 0.0800399950006249 Y_Plot3_Wheat + 0.0732365954255718 Y_Plot3_Rice
   + 0.0814406949131359 Y_Plot3_Corn + 0.1132565929258843 Y_Plot3_Soybeans
   + 0.0994496937882765 Y_Plot3_Lentils
   + 0.0822410948631421 Y_Plot3_Potatoes
   + 0.0908453943257093 Y_Plot3_Carrots
   + 0.108354143232096 Y_Plot3_Tomatoes + 0.0806402949631296 Y_Plot3_Apples
   + 0.0900449943757031 Y_Plot3_Oranges + 0.0834416947881515 Y_Plot3_Eggs
   + 0.0396450443694538 Y_Plot4_Wheat + 0.0362752155980502 Y_Plot4_Rice
   + 0.0403388326459193 Y_Plot4_Corn + 0.0560977377827772 Y_Plot4_Soybeans
   + 0.0492589676290464 Y_Plot4_Lentils
   + 0.0407352830896138 Y_Plot4_Potatoes
   + 0.0449971253593301 Y_Plot4_Carrots
   + 0.0536694788151481 Y_Plot4_Tomatoes
   + 0.0399423822022247 Y_Plot4_Apples + 0.0446006749156355 Y_Plot4_Oranges
   + 0.0413299587551556 Y_Plot4_Eggs + 0.0734408198975128 Y_Plot5_Wheat
   + 0.0671983502062242 Y_Plot5_Rice + 0.0747260342457193 Y_Plot5_Corn
   + 0.1039187601549806 Y_Plot5_Soybeans
   + 0.0912502187226597 Y_Plot5_Lentils
   + 0.0754604424446944 Y_Plot5_Potatoes
   + 0.083355330583677 Y_Plot5_Carrots + 0.099420509936258 Y_Plot5_Tomatoes
   + 0.0739916260467442 Y_Plot5_Apples + 0.0826209223847019 Y_Plot5_Oranges
   + 0.0765620547431571 Y_Plot5_Eggs + 0.0581427321584802 Y_Plot6_Wheat
   + 0.0532006 Y_Plot6_Rice + 0.0591602299712536 Y_Plot6_Corn
   + 0.0822719660042495 Y_Plot6_Soybeans
   + 0.0722423447069117 Y_Plot6_Lentils
   + 0.0597416572928384 Y_Plot6_Potatoes
   + 0.065992000999875 Y_Plot6_Carrots
   + 0.0787107236595426 Y_Plot6_Tomatoes
   + 0.0585788026496688 Y_Plot6_Apples + 0.0654105736782902 Y_Plot6_Oranges
   + 0.0606137982752156 Y_Plot6_Eggs + 0.0502437195350581 Y_Plot7_Wheat
   + 0.0459730033745782 Y_Plot7_Rice + 0.0511229846269216 Y_Plot7_Corn
   + 0.0710948631421073 Y_Plot7_Soybeans
   + 0.0624278215223097 Y_Plot7_Lentils
   + 0.0516254218222722 Y_Plot7_Potatoes
   + 0.057026621672291 Y_Plot7_Carrots
   + 0.0680174353205849 Y_Plot7_Tomatoes
   + 0.0506205474315711 Y_Plot7_Apples + 0.0565241844769404 Y_Plot7_Oranges
   + 0.0523790776152981 Y_Plot7_Eggs + 0.0263467066616673 Y_Plot8_Wheat
   + 0.0241072365954256 Y_Plot8_Rice + 0.0268077740282465 Y_Plot8_Corn
   + 0.0372805899262592 Y_Plot8_Soybeans
   + 0.0327357830271216 Y_Plot8_Lentils
   + 0.0270712410948631 Y_Plot8_Potatoes
   + 0.0299035120609924 Y_Plot8_Carrots
   + 0.0356668541432321 Y_Plot8_Tomatoes
   + 0.0265443069616298 Y_Plot8_Apples + 0.0296400449943757 Y_Plot8_Oranges
   + 0.0274664416947881 Y_Plot8_Eggs
Subject To
 Max_Assignment_Plot1: Y_Plot1_Wheat + Y_Plot1_Rice + Y_Plot1_Corn
   + Y_Plot1_Soybeans + Y_Plot1_Lentils + Y_Plot1_Potatoes
   + Y_Plot1_Carrots + Y_Plot1_Tomatoes + Y_Plot1_Apples + Y_Plot1_Oranges
   + Y_Plot1_Eggs <= 1
 Max_Assignment_Plot2: Y_Plot2_Wheat + Y_Plot2_Rice + Y_Plot2_Corn
   + Y_Plot2_Soybeans + Y_Plot2_Lentils + Y_Plot2_Potatoes
   + Y_Plot2_Carrots + Y_Plot2_Tomatoes + Y_Plot2_Apples + Y_Plot2_Oranges
   + Y_Plot2_Eggs <= 1
 Max_Assignment_Plot3: Y_Plot3_Wheat + Y_Plot3_Rice + Y_Plot3_Corn
   + Y_Plot3_Soybeans + Y_Plot3_Lentils + Y_Plot3_Potatoes
   + Y_Plot3_Carrots + Y_Plot3_Tomatoes + Y_Plot3_Apples + Y_Plot3_Oranges
   + Y_Plot3_Eggs <= 1
 Max_Assignment_Plot4: Y_Plot4_Wheat + Y_Plot4_Rice + Y_Plot4_Corn
   + Y_Plot4_Soybeans + Y_Plot4_Lentils + Y_Plot4_Potatoes
   + Y_Plot4_Carrots + Y_Plot4_Tomatoes + Y_Plot4_Apples + Y_Plot4_Oranges
   + Y_Plot4_Eggs <= 1
 Max_Assignment_Plot5: Y_Plot5_Wheat + Y_Plot5_Rice + Y_Plot5_Corn
   + Y_Plot5_Soybeans + Y_Plot5_Lentils + Y_Plot5_Potatoes
   + Y_Plot5_Carrots + Y_Plot5_Tomatoes + Y_Plot5_Apples + Y_Plot5_Oranges
   + Y_Plot5_Eggs <= 1
 Max_Assignment_Plot6: Y_Plot6_Wheat + Y_Plot6_Rice + Y_Plot6_Corn
   + Y_Plot6_Soybeans + Y_Plot6_Lentils + Y_Plot6_Potatoes
   + Y_Plot6_Carrots + Y_Plot6_Tomatoes + Y_Plot6_Apples + Y_Plot6_Oranges
   + Y_Plot6_Eggs <= 1
 Max_Assignment_Plot7: Y_Plot7_Wheat + Y_Plot7_Rice + Y_Plot7_Corn
   + Y_Plot7_Soybeans + Y_Plot7_Lentils + Y_Plot7_Potatoes
   + Y_Plot7_Carrots + Y_Plot7_Tomatoes + Y_Plot7_Apples + Y_Plot7_Oranges
   + Y_Plot7_Eggs <= 1
 Max_Assignment_Plot8: Y_Plot8_Wheat + Y_Plot8_Rice + Y_Plot8_Corn
   + Y_Plot8_Soybeans + Y_Plot8_Lentils + Y_Plot8_Potatoes
   + Y_Plot8_Carrots + Y_Plot8_Tomatoes + Y_Plot8_Apples + Y_Plot8_Oranges
   + Y_Plot8_Eggs <= 1
 U_Link_Plot1_Wheat: Y_Plot1_Wheat - U_Wheat <= 0
 U_Link_Plot2_Wheat: Y_Plot2_Wheat - U_Wheat <= 0
 U_Link_Plot3_Wheat: Y_Plot3_Wheat - U_Wheat <= 0
 U_Link_Plot4_Wheat: Y_Plot4_Wheat - U_Wheat <= 0
 U_Link_Plot5_Wheat: Y_Plot5_Wheat - U_Wheat <= 0
 U_Link_Plot6_Wheat: Y_Plot6_Wheat - U_Wheat <= 0
 U_Link_Plot7_Wheat: Y_Plot7_Wheat - U_Wheat <= 0
 U_Link_Plot8_Wheat: Y_Plot8_Wheat - U_Wheat <= 0
 U_Bound_Wheat: - Y_Plot1_Wheat - Y_Plot2_Wheat - Y_Plot3_Wheat
   - Y_Plot4_Wheat - Y_Plot5_Wheat - Y_Plot6_Wheat - Y_Plot7_Wheat
   - Y_Plot8_Wheat + U_Wheat <= 0
 U_Link_Plot1_Rice: Y_Plot1_Rice - U_Rice <= 0
 U_Link_Plot2_Rice: Y_Plot2_Rice - U_Rice <= 0
 U_Link_Plot3_Rice: Y_Plot3_Rice - U_Rice <= 0
 U_Link_Plot4_Rice: Y_Plot4_Rice - U_Rice <= 0
 U_Link_Plot5_Rice: Y_Plot5_Rice - U_Rice <= 0
 U_Link_Plot6_Rice: Y_Plot6_Rice - U_Rice <= 0
 U_Link_Plot7_Rice: Y_Plot7_Rice - U_Rice <= 0
 U_Link_Plot8_Rice: Y_Plot8_Rice - U_Rice <= 0
 U_Bound_Rice: - Y_Plot1_Rice - Y_Plot2_Rice - Y_Plot3_Rice - Y_Plot4_Rice
   - Y_Plot5_Rice - Y_Plot6_Rice - Y_Plot7_Rice - Y_Plot8_Rice + U_Rice
   <= 0
 U_Link_Plot1_Corn: Y_Plot1_Corn - U_Corn <= 0
 U_Link_Plot2_Corn: Y_Plot2_Corn - U_Corn <= 0
 U_Link_Plot3_Corn: Y_Plot3_Corn - U_Corn <= 0
 U_Link_Plot4_Corn: Y_Plot4_Corn - U_Corn <= 0
 U_Link_Plot5_Corn: Y_Plot5_Corn - U_Corn <= 0
 U_Link_Plot6_Corn: Y_Plot6_Corn - U_Corn <= 0
 U_Link_Plot7_Corn: Y_Plot7_Corn - U_Corn <= 0
 U_Link_Plot8_Corn: Y_Plot8_Corn - U_Corn <= 0
 U_Bound_Corn: - Y_Plot1_Corn - Y_Plot2_Corn - Y_Plot3_Corn - Y_Plot4_Corn
   - Y_Plot5_Corn - Y_Plot6_Corn - Y_Plot7_Corn - Y_Plot8_Corn + U_Corn
   <= 0
 U_Link_Plot1_Soybeans: Y_Plot1_Soybeans - U_Soybeans <= 0
 U_Link_Plot2_Soybeans: Y_Plot2_Soybeans - U_Soybeans <= 0
 U_Link_Plot3_Soybeans: Y_Plot3_Soybeans - U_Soybeans <= 0
 U_Link_Plot4_Soybeans: Y_Plot4_Soybeans - U_Soybeans <= 0
 U_Link_Plot5_Soybeans: Y_Plot5_Soybeans - U_Soybeans <= 0
 U_Link_Plot6_Soybeans: Y_Plot6_Soybeans - U_Soybeans <= 0
 U_Link_Plot7_Soybeans: Y_Plot7_Soybeans - U_Soybeans <= 0
 U_Link_Plot8_Soybeans: Y_Plot8_Soybeans - U_Soybeans <= 0
 U_Bound_Soybeans: - Y_Plot1_Soybeans - Y_Plot2_Soybeans - Y_Plot3_Soybeans
   - Y_Plot4_Soybeans - Y_Plot5_Soybeans - Y_Plot6_Soybeans
   - Y_Plot7_Soybeans - Y_Plot8_Soybeans + U_Soybeans <= 0
 U_Link_Plot1_Lentils: Y_Plot1_Lentils - U_Lentils <= 0
 U_Link_Plot2_Lentils: Y_Plot2_Lentils - U_Lentils <= 0
 U_Link_Plot3_Lentils: Y_Plot3_Lentils - U_Lentils <= 0
 U_Link_Plot4_Lentils: Y_Plot4_Lentils - U_Lentils <= 0
 U_Link_Plot5_Lentils: Y_Plot5_Lentils - U_Lentils <= 0
 U_Link_Plot6_Lentils: Y_Plot6_Lentils - U_Lentils <= 0
 U_Link_Plot7_Lentils: Y_Plot7_Lentils - U_Lentils <= 0
 U_Link_Plot8_Lentils: Y_Plot8_Lentils - U_Lentils <= 0
 U_Bound_Lentils: - Y_Plot1_Lentils - Y_Plot2_Lentils - Y_Plot3_Lentils
   - Y_Plot4_Lentils - Y_Plot5_Lentils - Y_Plot6_Lentils - Y_Plot7_Lentils
   - Y_Plot8_Lentils + U_Lentils <= 0
 U_Link_Plot1_Potatoes: Y_Plot1_Potatoes - U_Potatoes <= 0
 U_Link_Plot2_Potatoes: Y_Plot2_Potatoes - U_Potatoes <= 0
 U_Link_Plot3_Potatoes: Y_Plot3_Potatoes - U_Potatoes <= 0
 U_Link_Plot4_Potatoes: Y_Plot4_Potatoes - U_Potatoes <= 0
 U_Link_Plot5_Potatoes: Y_Plot5_Potatoes - U_Potatoes <= 0
 U_Link_Plot6_Potatoes: Y_Plot6_Potatoes - U_Potatoes <= 0
 U_Link_Plot7_Potatoes: Y_Plot7_Potatoes - U_Potatoes <= 0
 U_Link_Plot8_Potatoes: Y_Plot8_Potatoes - U_Potatoes <= 0
 U_Bound_Potatoes: - Y_Plot1_Potatoes - Y_Plot2_Potatoes - Y_Plot3_Potatoes
   - Y_Plot4_Potatoes - Y_Plot5_Potatoes - Y_Plot6_Potatoes
   - Y_Plot7_Potatoes - Y_Plot8_Potatoes + U_Potatoes <= 0
 U_Link_Plot1_Carrots: Y_Plot1_Carrots - U_Carrots <= 0
 U_Link_Plot2_Carrots: Y_Plot2_Carrots - U_Carrots <= 0
 U_Link_Plot3_Carrots: Y_Plot3_Carrots - U_Carrots <= 0
 U_Link_Plot4_Carrots: Y_Plot4_Carrots - U_Carrots <= 0
 U_Link_Plot5_Carrots: Y_Plot5_Carrots - U_Carrots <= 0
 U_Link_Plot6_Carrots: Y_Plot6_Carrots - U_Carrots <= 0
 U_Link_Plot7_Carrots: Y_Plot7_Carrots - U_Carrots <= 0
 U_Link_Plot8_Carrots: Y_Plot8_Carrots - U_Carrots <= 0
 U_Bound_Carrots: - Y_Plot1_Carrots - Y_Plot2_Carrots - Y_Plot3_Carrots
   - Y_Plot4_Carrots - Y_Plot5_Carrots - Y_Plot6_Carrots - Y_Plot7_Carrots
   - Y_Plot8_Carrots + U_Carrots <= 0
 U_Link_Plot1_Tomatoes: Y_Plot1_Tomatoes - U_Tomatoes <= 0
 U_Link_Plot2_Tomatoes: Y_Plot2_Tomatoes - U_Tomatoes <= 0
 U_Link_Plot3_Tomatoes: Y_Plot3_Tomatoes - U_Tomatoes <= 0
 U_Link_Plot4_Tomatoes: Y_Plot4_Tomatoes - U_Tomatoes <= 0
 U_Link_Plot5_Tomatoes: Y_Plot5_Tomatoes - U_Tomatoes <= 0
 U_Link_Plot6_Tomatoes: Y_Plot6_Tomatoes - U_Tomatoes <= 0
 U_Link_Plot7_Tomatoes: Y_Plot7_Tomatoes - U_Tomatoes <= 0
 U_Link_Plot8_Tomatoes: Y_Plot8_Tomatoes - U_Tomatoes <= 0
 U_Bound_Tomatoes: - Y_Plot1_Tomatoes - Y_Plot2_Tomatoes - Y_Plot3_Tomatoes
   - Y_Plot4_Tomatoes - Y_Plot5_Tomatoes - Y_Plot6_Tomatoes
   - Y_Plot7_Tomatoes - Y_Plot8_Tomatoes + U_Tomatoes <= 0
 U_Link_Plot1_Apples: Y_Plot1_Apples - U_Apples <= 0
 U_Link_Plot2_Apples: Y_Plot2_Apples - U_Apples <= 0
 U_Link_Plot3_Apples: Y_Plot3_Apples - U_Apples <= 0
 U_Link_Plot4_Apples: Y_Plot4_Apples - U_Apples <= 0
 U_Link_Plot5_Apples: Y_Plot5_Apples - U_Apples <= 0
 U_Link_Plot6_Apples: Y_Plot6_Apples - U_Apples <= 0
 U_Link_Plot7_Apples: Y_Plot7_Apples - U_Apples <= 0
 U_Link_Plot8_Apples: Y_Plot8_Apples - U_Apples <= 0
 U_Bound_Apples: - Y_Plot1_Apples - Y_Plot2_Apples - Y_Plot3_Apples
   - Y_Plot4_Apples - Y_Plot5_Apples - Y_Plot6_Apples - Y_Plot7_Apples
   - Y_Plot8_Apples + U_Apples <= 0
 U_Link_Plot1_Oranges: Y_Plot1_Oranges - U_Oranges <= 0
 U_Link_Plot2_Oranges: Y_Plot2_Oranges - U_Oranges <= 0
 U_Link_Plot3_Oranges: Y_Plot3_Oranges - U_Oranges <= 0
 U_Link_Plot4_Oranges: Y_Plot4_Oranges - U_Oranges <= 0
 U_Link_Plot5_Oranges: Y_Plot5_Oranges - U_Oranges <= 0
 U_Link_Plot6_Oranges: Y_Plot6_Oranges - U_Oranges <= 0
 U_Link_Plot7_Oranges: Y_Plot7_Oranges - U_Oranges <= 0
 U_Link_Plot8_Oranges: Y_Plot8_Oranges - U_Oranges <= 0
 U_Bound_Oranges: - Y_Plot1_Oranges - Y_Plot2_Oranges - Y_Plot3_Oranges
   - Y_Plot4_Oranges - Y_Plot5_Oranges - Y_Plot6_Oranges - Y_Plot7_Oranges
   - Y_Plot8_Oranges + U_Oranges <= 0
 U_Link_Plot1_Eggs: Y_Plot1_Eggs - U_Eggs <= 0
 U_Link_Plot2_Eggs: Y_Plot2_Eggs - U_Eggs <= 0
 U_Link_Plot3_Eggs: Y_Plot3_Eggs - U_Eggs <= 0
 U_Link_Plot4_Eggs: Y_Plot4_Eggs - U_Eggs <= 0
 U_Link_Plot5_Eggs: Y_Plot5_Eggs - U_Eggs <= 0
 U_Link_Plot6_Eggs: Y_Plot6_Eggs - U_Eggs <= 0
 U_Link_Plot7_Eggs: Y_Plot7_Eggs - U_Eggs <= 0
 U_Link_Plot8_Eggs: Y_Plot8_Eggs - U_Eggs <= 0
 U_Bound_Eggs: - Y_Plot1_Eggs - Y_Plot2_Eggs - Y_Plot3_Eggs - Y_Plot4_Eggs
   - Y_Plot5_Eggs - Y_Plot6_Eggs - Y_Plot7_Eggs - Y_Plot8_Eggs + U_Eggs
   <= 0
 MinFoodGroup_Grains: U_Wheat + U_Rice + U_Corn >= 1
 MinFoodGroup_Legumes: U_Soybeans + U_Lentils >= 1
 MinFoodGroup_Vegetables: U_Potatoes + U_Carrots + U_Tomatoes >= 1
 MinFoodGroup_Fruits: U_Apples + U_Oranges >= 1
 MinFoodGroup_Proteins: U_Eggs >= 1
Bounds
Binaries
 Y_Plot1_Wheat Y_Plot1_Rice Y_Plot1_Corn Y_Plot1_Soybeans Y_Plot1_Lentils
 Y_Plot1_Potatoes Y_Plot1_Carrots Y_Plot1_Tomatoes Y_Plot1_Apples
 Y_Plot1_Oranges Y_Plot1_Eggs Y_Plot2_Wheat Y_Plot2_Rice Y_Plot2_Corn
 Y_Plot2_Soybeans Y_Plot2_Lentils Y_Plot2_Potatoes Y_Plot2_Carrots
 Y_Plot2_Tomatoes Y_Plot2_Apples Y_Plot2_Oranges Y_Plot2_Eggs Y_Plot3_Wheat
 Y_Plot3_Rice Y_Plot3_Corn Y_Plot3_Soybeans Y_Plot3_Lentils
 Y_Plot3_Potatoes Y_Plot3_Carrots Y_Plot3_Tomatoes Y_Plot3_Apples
 Y_Plot3_Oranges Y_Plot3_Eggs Y_Plot4_Wheat Y_Plot4_Rice Y_Plot4_Corn
 Y_Plot4_Soybeans Y_Plot4_Lentils Y_Plot4_Potatoes Y_Plot4_Carrots
 Y_Plot4_Tomatoes Y_Plot4_Apples Y_Plot4_Oranges Y_Plot4_Eggs Y_Plot5_Wheat
 Y_Plot5_Rice Y_Plot5_Corn Y_Plot5_Soybeans Y_Plot5_Lentils
 Y_Plot5_Potatoes Y_Plot5_Carrots Y_Plot5_Tomatoes Y_Plot5_Apples
 Y_Plot5_Oranges Y_Plot5_Eggs Y_Plot6_Wheat Y_Plot6_Rice Y_Plot6_Corn
 Y_Plot6_Soybeans Y_Plot6_Lentils Y_Plot6_Potatoes Y_Plot6_Carrots
 Y_Plot6_Tomatoes Y_Plot6_Apples Y_Plot6_Oranges Y_Plot6_Eggs Y_Plot7_Wheat
 Y_Plot7_Rice Y_Plot7_Corn Y_Plot7_Soybeans Y_Plot7_Lentils
 Y_Plot7_Potatoes Y_Plot7_Carrots Y_Plot7_Tomatoes Y_Plot7_Apples
 Y_Plot7_Oranges Y_Plot7_Eggs Y_Plot8_Wheat Y_Plot8_Rice Y_Plot8_Corn
 Y_Plot8_Soybeans Y_Plot8_Lentils Y_Plot8_Potatoes Y_Plot8_Carrots
 Y_Plot8_Tomatoes Y_Plot8_Apples Y_Plot8_Oranges Y_Plot8_Eggs U_Wheat
 U_Rice U_Corn U_Soybeans U_Lentils U_Potatoes U_Carrots U_Tomatoes
 U_Apples U_Oranges U_Eggs
End
