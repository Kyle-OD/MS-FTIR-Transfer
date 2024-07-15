import json
from typing import Dict, List, Tuple
import csv

def extract_compound_data(compound: Dict) -> Dict:
    result = {
        'kind': compound.get('kind', ''),
        'name': compound.get('names', [{}])[0].get('name', '') if compound.get('names') else '',
        'molecular_formula': '',
        'SMILES': '',
        'pubchem_cid': '',
        'InChI': compound.get('inchi', ''),
        'InChIKey': compound.get('inchiKey', ''),
        'total_exact_mass': ''
    }
    
    for meta in compound.get('metaData', []):
        if meta['name'] == 'molecular formula':
            result['molecular_formula'] = meta.get('value', '')
        elif meta['name'] == 'SMILES':
            result['SMILES'] = meta.get('value', '')
        elif meta['name'] == 'total exact mass':
            result['total_exact_mass'] = meta.get('value', '')
        elif meta['name'] == 'pubchem cid':
            result['pubchem_cid'] = meta.get('value', '')
    
    return result

def extract_metadata(data: Dict) -> Dict:
    result = {
        'exact_mass': '',
        'instrument': '',
        'instrument_type': '',
        'ms_level': '',
        'ionization': '',
        'ionization_mode': ''
    }
    
    for meta in data.get('metaData', []):
        if meta['name'] == 'precursor m/z':
            result['exact_mass'] = meta.get('value', '')
        elif meta['name'] == 'instrument':
            result['instrument'] = meta.get('value', '')
        elif meta['name'] == 'instrumenttype':
            result['instrument_type'] = meta.get('value', '')
        elif meta['name'] == 'ms level':
            result['ms_level'] = meta.get('value', '')
        elif meta['name'] == 'ionization':
            result['ionization'] = meta.get('value', '')
        elif meta['name'] == 'ionization mode':
            result['ionization_mode'] = meta.get('value', '')
    
    return result

def parse_spectrum(spectrum: str) -> List[Tuple[float, float]]:
    result = []
    for pair in spectrum.split():
        try:
            mass, abundance = pair.split(':')
            mass = float(mass.strip())
            abundance = float(abundance.strip())
            if mass and abundance:  # Ensure both values are non-zero
                result.append((mass, abundance))
        except ValueError:
            # Skip this pair if there's any issue converting to float
            continue
    return result

def process_json(json_data: Dict) -> Dict:
    result = {}
    
    if 'compound' in json_data and json_data['compound']:
        result.update(extract_compound_data(json_data['compound'][0]))
    
    result.update(extract_metadata(json_data))
    
    result['spectrum'] = parse_spectrum(json_data.get('spectrum', ''))
    result['score'] = json_data.get('score', {}).get('score', '')
    
    return result

def process_json_file(file_path: str, output_csv: str):
    with open(file_path, 'r') as f:
        json_data = json.load(f)
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['kind', 'name', 'molecular_formula', 'SMILES', 'pubchem_cid', 'InChI', 'InChIKey', 
                         'total_exact_mass', 'exact_mass', 'instrument', 'instrument_type', 
                         'ms_level', 'ionization', 'ionization_mode', 'spectrum', 'score'])
        
        if isinstance(json_data, list):
            for item in json_data:
                processed_data = process_json(item)
                writer.writerow(list(processed_data.values()))
        elif isinstance(json_data, dict):
            processed_data = process_json(json_data)
            writer.writerow(list(processed_data.values()))
        else:
            raise ValueError("Unexpected JSON structure")

# Usage
#process_json_file('input.json', 'output.csv')
