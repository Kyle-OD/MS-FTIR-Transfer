from MoNA_reader import process_json_file

json_in_silico1 = "data/MoNA/MoNA-export-In-Silico_Spectra-json/MoNA-export-In-Silico_Spectra_pt1.json"
json_in_silico2 = "data/MoNA/MoNA-export-In-Silico_Spectra-json/MoNA-export-In-Silico_Spectra_pt2.json"
json_in_silico3 = "data/MoNA/MoNA-export-In-Silico_Spectra-json/MoNA-export-In-Silico_Spectra_pt3.json"
json_in_silico4 = "data/MoNA/MoNA-export-In-Silico_Spectra-json/MoNA-export-In-Silico_Spectra_pt4.json"
json_in_silico5 = "data/MoNA/MoNA-export-In-Silico_Spectra-json/MoNA-export-In-Silico_Spectra_pt5.json"
json_in_silico6 = "data/MoNA/MoNA-export-In-Silico_Spectra-json/MoNA-export-In-Silico_Spectra_pt6.json"
json_in_silico7 = "data/MoNA/MoNA-export-In-Silico_Spectra-json/MoNA-export-In-Silico_Spectra_pt7.json"
json_in_silico8 = "data/MoNA/MoNA-export-In-Silico_Spectra-json/MoNA-export-In-Silico_Spectra_pt8.json"
json_experimental = "data/MoNA/MoNA-export-Experimental_Spectra-json/MoNA-export-Experimental_Spectra.json"

csv_experimental = "data/MoNA/experimental.csv"
csv_in_silico1 = "data/MoNA/in-silico1.csv"
csv_in_silico2 = "data/MoNA/in-silico2.csv"
csv_in_silico3 = "data/MoNA/in-silico3.csv"
csv_in_silico4 = "data/MoNA/in-silico4.csv"
csv_in_silico5 = "data/MoNA/in-silico5.csv"
csv_in_silico6 = "data/MoNA/in-silico6.csv"
csv_in_silico7 = "data/MoNA/in-silico7.csv"
csv_in_silico8 = "data/MoNA/in-silico8.csv"


# process experimental data
print("Starting Experimental Extract")
process_json_file(json_experimental, csv_experimental)
print("Finished Experimental Extract")

# process in-silico data
print("Starting In-Silico 1 Extract")
process_json_file(json_in_silico1, csv_in_silico1)
print("Finished In-Silico 1 Extract")

print("Starting In-Silico 2 Extract")
process_json_file(json_in_silico2, csv_in_silico2)
print("Finished In-Silico 2 Extract")

print("Starting In-Silico 3 Extract")
process_json_file(json_in_silico3, csv_in_silico3)
print("Finished In-Silico 3 Extract")

print("Starting In-Silico 4 Extract")
process_json_file(json_in_silico4, csv_in_silico4)
print("Finished In-Silico 4 Extract")

print("Starting In-Silico 5 Extract")
process_json_file(json_in_silico5, csv_in_silico5)
print("Finished In-Silico 5 Extract")

print("Starting In-Silico 6 Extract")
process_json_file(json_in_silico6, csv_in_silico6)
print("Finished In-Silico 6 Extract")

print("Starting In-Silico 7 Extract")
process_json_file(json_in_silico7, csv_in_silico7)
print("Finished In-Silico 7 Extract")

print("Starting In-Silico 8 Extract")
process_json_file(json_in_silico8, csv_in_silico8)
print("Finished In-Silico 8 Extract")
