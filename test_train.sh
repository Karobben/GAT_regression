echo "pdb_path,y,antibody_chains,antigen_chains" > manifest.csv
grep 7KC1_21_1 ../../DataClean/result/Final_antibody_antigen_pairs_experimental_data.tsv| awk -F"\t" '{OFS=","; print $12,$5,"\""$13"\"","\""$14"\""}' >> manifest.csv

python ../train.py --manifest manifest.csv







echo "pdb_path,y,antibody_chains,antigen_chains" > manifest2.csv
grep 5K9J_16_1 ../HA_DB/Analysis/DataClean/result/Final_antibody_antigen_pairs_experimental_data.tsv| awk -F"\t" '{OFS=","; print $12,$5,"\""$13"\"","\""$14"\""}'| sed 's/>50/100/' >> manifest2.csv

 
