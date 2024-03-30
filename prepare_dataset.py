import os
import json
import random
import psycopg2
import pandas as pd
from argparse import ArgumentParser

LABEVENT_CATEGORIES_QUERY = """
SELECT itemid, label, fluid, category, loinc_code
FROM mimiciii.d_labitems order by category, fluid, label;
"""

LABEVENTS_QUERY = """
SELECT
    a.subject_id,
    a.hadm_id,
    a.admittime,

    -- Patient demographics
    a.insurance,
	a.religion,
	a.marital_status,
	a.ethnicity,
	
    le.itemid,
	le.charttime,
	le.value,
	le.valueuom,
	le.flag,
	
	li.label,
	li.fluid,
	li.category,
	li.loinc_code
FROM mimiciii.admissions a
INNER JOIN mimiciii.labevents le 
	ON a.subject_id = le.subject_id
		AND a.hadm_id = le.hadm_id
INNER JOIN mimiciii.d_labitems li
	ON le.itemid = li.itemid limit 500000;
"""

NOTES_QUERY = """
SELECT * FROM mimiciii.noteevents WHERE charttime IS NOT NULL;
"""


def get_db_connection():
    user = os.environ.get('MIMIC_USER') or 'postgres'
    password = os.environ.get('MIMIC_PASSWORD') or 'testing'

    conn_string = f"host='localhost' dbname='mimic' user='{user}' password='{password}'"
    conn = psycopg2.connect(conn_string)
    return conn

def prepare_dataset(training_size, validation_size, test_size):
    # Create dataset directory if it does not exist
    if not os.path.exists('data'):
        os.makedirs('data')

    with get_db_connection() as conn:
        # Get unique labevents - these are our labels
        labevent_categories_df = pd.read_sql(LABEVENT_CATEGORIES_QUERY, conn)
        labevent_categories = []
        for _, labevent in labevent_categories_df.iterrows():
            labevent_categories.append({
                'itemid': labevent['itemid'],
                'label': labevent['label'],
                'fluid': labevent['fluid'],
                'category': labevent['category'],
                'loinc_code': labevent['loinc_code']
            })

        label_map = {labevent['itemid']: i for i, labevent in enumerate(labevent_categories)}

        label_fp = 'data/label.json'
        with open(label_fp, 'w') as f:
            json.dump(labevent_categories, f)

        # Get labevents data
        labevents_df = pd.read_sql(LABEVENTS_QUERY, conn)
        noteevents_df = pd.read_sql(NOTES_QUERY, conn)

        labevent_records = preprocess_dataset(labevents_df, noteevents_df, label_map, size = training_size + validation_size + test_size)

        # shuffle the records
        
        random.shuffle(labevent_records)

        training_records = labevent_records[:training_size]
        validation_records = labevent_records[training_size:training_size + validation_size]
        test_records = labevent_records[training_size + validation_size:training_size + validation_size + test_size]

        training_fp = 'data/training.json'
        validation_fp = 'data/validation.json'
        test_fp = 'data/test.json'

        with open(training_fp, 'w') as f:
            json.dump(training_records, f)

        with open(validation_fp, 'w') as f:
            json.dump(validation_records, f)

        with open(test_fp, 'w') as f:
            json.dump(test_records, f)

        print(f"Training records: {len(training_records)}")
        print(f"Validation records: {len(validation_records)}")
        print(f"Test records: {len(test_records)}")


def preprocess_dataset(labevents_df, noteevents_df, label_map,  size):
    records = []

    # get unique admissions and subjects
    admissions = labevents_df[['subject_id', 'hadm_id', 'admittime', 'insurance', 'religion', 'marital_status', 'ethnicity']].drop_duplicates()
    
    for _, admission in admissions.iterrows():

        # get labevents for this admission
        admission_labevents = labevents_df[(labevents_df['subject_id'] == admission['subject_id']) & (labevents_df['hadm_id'] == admission['hadm_id'])]
        admission_noteevents = noteevents_df[(noteevents_df['subject_id'] == admission['subject_id']) & (noteevents_df['hadm_id'] == admission['hadm_id'])]

        # for each noteevent, get the list of labevents that occurr after the noteevent

        for _, note in admission_noteevents.iterrows():
            note_time = note['charttime']
            if note_time:
                # get all notes that occured before the note
                previous_notes = admission_noteevents[admission_noteevents['charttime'] < note_time]
                note_text = ' '.join(previous_notes['text']) + ' ' + note['text']

                # get labevents that occur after the note
                previous_note_labevents = admission_labevents[admission_labevents['charttime'] < note_time]
                upcoming_note_labevents = admission_labevents[admission_labevents['charttime'] > note_time]
                if len(upcoming_note_labevents) > 0:
                    prev_labels = list(set([ label_map[itemid] for itemid in previous_note_labevents['itemid'] ]))
                    prev_labels = sorted(prev_labels)

                    # get the label for each labevent
                    upcoming_labels = list(set([ label_map[itemid] for itemid in upcoming_note_labevents['itemid'] ]))
                    upcoming_labels = sorted(upcoming_labels)

                    records.append({
                        'subject_id': admission['subject_id'],
                        'hadm_id': admission['hadm_id'],
                        'insurance': admission['insurance'],
                        'religion': admission['religion'],
                        'marital_status': admission['marital_status'],
                        'ethnicity': admission['ethnicity'],
                        'note_text': note_text,
                        'prev_labels': prev_labels,
                        'labels': upcoming_labels
                    })

                    if len(records) >= size:
                        return records
    return records


    


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--training_size", type=int, default=10000)
    parser.add_argument("--validation_size", type=int, default=1000)
    parser.add_argument("--test_size", type=int, default=1000)
    args = parser.parse_args()

    training_size = args.training_size
    validation_size = args.validation_size
    test_size = args.test_size

    prepare_dataset(training_size, validation_size, test_size)