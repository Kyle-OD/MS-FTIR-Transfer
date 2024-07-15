from readspc import read, spcreader

hdr, subhdr, x, y = read("PNNL FTIR Data/Proposal 67367/Bruker Spectrometer/DATABASE-SERDP-4-0-Aug-2018-Acetol-Croto/Acrylic acid/ACRACID_50T.SPC")

print(hdr)
print(subhdr)
print(x)
print(y)
