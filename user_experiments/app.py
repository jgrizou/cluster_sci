from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, session

from datetime import datetime
import random
import os
import csv

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


import sys
sys.path.append(os.path.join('..', 'tools'))
import saving_tools
import file_tools
_FACE_DIR = os.path.join('.', 'static')
_RESULT_DIR = os.path.join('.', 'results')
file_tools.ensure_dir(_RESULT_DIR)

_BLANK_FACE_PATH = 'static/default_face.jpg'
_SHOWING_TIMES_MS = [500]
_MAX_NUMBER_OF_TEST = 30

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Add this line for session management


@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/experiment')
def index():
    username = session.get('username', 'unknown')
    return render_template('index.html', username=username)

@app.route('/start_experiment', methods=['POST'])
def start_experiment():
    username = request.form.get('username')
    if username:
        session['username'] = username
        return redirect(url_for('index'))
    return redirect(url_for('welcome'))

@app.route('/get_faces')
def get_faces():
    all_face_folders = list(file_tools.list_folders(_FACE_DIR))
    selected_face_folder = random.choice(all_face_folders)
    
    info_filename = os.path.join(selected_face_folder, 'info.json')
    info = saving_tools.load_dict_from_json(info_filename)

    target_image_path = os.path.join(selected_face_folder, 'target.jpg')
    target_image_path = file_tools.change_refpath(target_image_path, HERE_PATH, '.')

    test_image_path = os.path.join(selected_face_folder, 'test.jpg')
    test_image_path = file_tools.change_refpath(test_image_path, HERE_PATH, '.')

    display_choices = [target_image_path, test_image_path]
    random.shuffle(display_choices)

    username = session.get('username', 'unknown')
    experiment_count = get_experiment_count(username)
    
    return jsonify({
        'target': target_image_path,
        'choices': display_choices,
        'default': _BLANK_FACE_PATH,
        'showing_time': random.choice(_SHOWING_TIMES_MS),
        'rounded_d': info['rounded_d'], 
        'experiment_count': experiment_count
    })

@app.route('/save_result', methods=['POST'])
def save_result():
    data = request.json
    username = session.get('username', 'unknown')
    filename = os.path.join(_RESULT_DIR, f'{username}.csv') 

    current_timestamp = datetime.now()
    csv_friendly_timestamp = current_timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
    # Check if the file exists to determine if we need to write the header
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'target', 'chosen', 'showing_time', 'rounded_d'])  # Write header if file is new
        writer.writerow([csv_friendly_timestamp, data['target'], data['chosen'], data['showing_time'], data['rounded_d']])
    
    experiment_count = get_experiment_count(username)

    if experiment_count >= 30:
        return jsonify({'status': 'complete', 'experiment_count': experiment_count})
    else:
        return jsonify({'status': 'success', 'experiment_count': experiment_count})

def get_experiment_count(username):
    filename = os.path.join(_RESULT_DIR, f'{username}.csv')
    if not os.path.isfile(filename):
        return 0
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        return sum(1 for row in reader) - 1  # Subtract 1 to account for header row

@app.route('/experiment_complete')
def experiment_complete():
    username = session.get('username', 'unknown')
    experiment_count = get_experiment_count(username)
    return render_template('experiment_complete.html', experiment_count=experiment_count)



if __name__ == '__main__':
    app.run(debug=True)