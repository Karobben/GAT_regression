echo "pdb_path,y,antibody_chains,antigen_chains" > manifest.csv
grep 7KC1_21_1 ../../DataClean/result/Final_antibody_antigen_pairs_experimental_data.tsv| awk -F"\t" '{OFS=","; print $12,$5,"\""$13"\"","\""$14"\""}' >> manifest.csv

python train.py --manifest manifest.csv
