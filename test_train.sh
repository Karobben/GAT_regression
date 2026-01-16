# make a test dataset 
echo "pdb_path,y,antibody_chains,antigen_chains" > manifest.csv
grep 7KC1_21_1 ../../DataClean/result/Final_antibody_antigen_pairs_experimental_data.tsv| awk -F"\t" '{OFS=","; print $12,$5,"\""$13"\"","\""$14"\""}' >> manifest.csv

python ../train.py --manifest manifest.csv


# make the second test dataset
echo "pdb_path,y,antibody_chains,antigen_chains" > manifest2.csv
grep 5K9J_16_1 ../HA_DB/Analysis/DataClean/result/Final_antibody_antigen_pairs_experimental_data.tsv| awk -F"\t" '{OFS=","; print $12,$5,"\""$13"\"","\""$14"\""}'| sed 's/>50/100/' >> manifest2.csv

# Make the binary dataset 
echo "pdb_path,y,antibody_chains,antigen_chains" > train_set/binary_manifest.csv
cat ../HA_DB/Analysis/DataClean/result/Final_antibody_antigen_pairs_experimental_data.tsv| awk -F"\t" '{OFS=","; print $12,$6,"\""$13"\"","\""$14"\""}'| sed 's/>50/100/' >> train_set/binary_manifest.csv
sed -i '/9DRU_SH52A/d' train_set/binary_manifest.csv  # remove header
sed -i '/is_neutralized/d' train_set/binary_manifest.csv  # remove test data
sed -i '/3LZF_DA187E_DA222G.pdb/d' train_set/binary_manifest.csv  # remove test data
sed -i '/5UG0_MUT.pdb/d' train_set/binary_manifest.csv  # remove test data
sed -i '/5UMN_/d' train_set/binary_manifest.csv  # remove test data
sed -i '/4FQR_4FQR.pdb/d' train_set/binary_manifest.csv  # remove test data
sed -i '/6D0U_/d' train_set/binary_manifest.csv  # remove test data
sed -i '/7KQG_M/d' train_set/binary_manifest.csv  # remove test data
sed -i '/7KQG_D/d' train_set/binary_manifest.csv  # remove test data
sed -i '/8VEB_/d' train_set/binary_manifest.csv  # remove test data
sed -i '/9DRU_/d' train_set/binary_manifest.csv  # remove test data
sed -i '/9DS1_/d' train_set/binary_manifest.csv  # remove test data
sed -i '/9DS2_/d' train_set/binary_manifest.csv  # remove test data
sed -i '/9DM0_/d' train_set/binary_manifest.csv  # remove test data
sed -i '/7T3D_/d' train_set/binary_manifest.csv  # remove test data












 
