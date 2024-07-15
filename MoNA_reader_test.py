import json
import subprocess
import csv
from typing import Dict, List, Tuple, Optional
from decimal import Decimal

def extract_compound_data(compound: Dict) -> Dict:
    result = {
        'kind': compound.get('kind', ''),
        'name': compound.get('names', [{}])[0].get('name', '') if compound.get('names') else '',
        'molecular_formula': '',
        'pubchem_cid': '',
        'InChI': compound.get('inchi', ''),
        'InChIKey': compound.get('inchiKey', ''),
        'total_exact_mass': ''
    }
    
    meta = compound.get('metaData', [])
    for meta in compound.get('metaData', []):
        if meta['name'] == 'molecular formula':
            result['molecular_formula'] = meta.get('value', '')
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

def read_line(file_path: str, line_number: int):
    json_entry = subprocess.run(['sed', '-n', str(line_number)+'p;'+str(line_number+1)+'q', file_path], capture_output=True).stdout[0:-2]
    return json_entry

def process_json_file(file_path: str, output_csv: str, total_lines: int=None):
    with open(file_path, 'rb') as json_file, open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['kind', 'name', 'molecular_formula', 'pubchem_cid', 'InChI', 'InChIKey', 
                         'total_exact_mass', 'exact_mass', 'instrument', 'instrument_type', 
                         'ms_level', 'ionization', 'ionization_mode', 'spectrum', 'score'])

        current_item = {}
        item_count = 0
        if total_lines is None:
            print('Reading lines...')
            total_lines = int(subprocess.run(['wc', '-l', file_path], capture_output=True, text=True).stdout.split(' ')[0])
        print('Lines in file:', total_lines)
        for line in range(2, total_lines):
            current_item = json.loads(read_line(file_path, line))
            try:
                processed_data = process_json(current_item)
                writer.writerow(list(processed_data.values()))
                item_count += 1
                if item_count % 1000 == 0:
                    print(f"Processed {item_count} items")
            except Exception as e:
                print(f"Error processing item: {e}")
                print(f"Problematic item: {current_item}")
                current_item = {}

        print(f"Total items processed: {item_count}")
