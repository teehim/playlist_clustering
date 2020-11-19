from flask import Flask, render_template, make_response, redirect, url_for, request, session, jsonify

import pickle
import math
import requests
from urllib.parse import urlencode
from config import DefaultConfig
from datetime import datetime, timedelta
from flask_cors import CORS

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

CONFIG = DefaultConfig()

app = Flask(__name__)
CORS(app)

app.secret_key = b'2aa45112fa0e4d38befc0cbde25006e1'
client_id = 'ca83153c363a4fbc8fc683647b84831d'
client_secret = '2aa45112fa0e4d38befc0cbde25006e1'
redirect_uri = 'https://playlist-clustering.herokuapp.com'

letters = 'abcdefghijklmnopqrstuvwxyz0123456789'
stateKey = 'spotify_auth_state'
import base64

@app.route('/login', methods=['POST'])
def login():
    access_token = request.json['access_token']
    headers = {
        'Authorization': 'Bearer ' + access_token
    }
    rme = requests.get('https://api.spotify.com/v1/me', headers=headers)
    user = rme.json()
    rplaylists = requests.get('https://api.spotify.com/v1/me/playlists', headers=headers)
    playlists = rplaylists.json()['items']
    playlist_items = []
    for playlist in playlists:
        playlist_items.append({
            'id': playlist["id"],
            'name': playlist["name"],
            'images': playlist["images"],
            'track_counts': playlist['tracks']['total']
        })

    user['playlists'] = playlist_items

    return jsonify(user)


@app.route('/create_playlist')
def create_playlist():
    # access_token = get_service_token()
    headers = {
        'Authorization': 'Bearer ' + 'BQC5jji4F0NmGb109bmhfAXZ8O7qR0u5BVz3_huawvti_Y4wQ1p_PQ5Q4zG0jtXuF0oN3tPkfrA8DGz4lL1hH-56Ih7SvvLa7F1FGGhrfefg7FeCnUBv22ScYe60lINu7GKnHV-M8JdUvV-i6Rrd_qR9bnLtFDUEMBIHiugi31RTSYjah4D4hDrPCnHt1GZCnLZ2Ms97ijs4I-j4EDWVYA'
    }
    track_ids = ['spotify:track:2LeLsYwDTlfvWfqgIbxvgG',
       'spotify:track:0PACdf6vCQee3ICq02EWlf',
       'spotify:track:5qcwzdhDQjHjmu12gOLjY0',
       'spotify:track:3p7byaSR5J1he8vvDkxirp',
       'spotify:track:0AqMAXSzkIRMV0alAddJK8',
       'spotify:track:3TmMHzw7592o4bwtrSHeYv',
       'spotify:track:4J3eFXXi2nQqF6MWLprnbR',
       'spotify:track:2Fxmhks0bxGSBdJ92vM42m',
       'spotify:track:6IRdLKIyS4p7XNiP8r6rsx',
       'spotify:track:3Tc57t9l2O8FwQZtQOvPXK',
       'spotify:track:218AUAkMEZWsVwjPxs80OI',
       'spotify:track:2mosyIQjuejKlqSsdoI6q4',
       'spotify:track:2ylVfK4pVfeSV4zxieyT2B',
       'spotify:track:1ZI0AGxSFv2bW3STGNDhB7',
       'spotify:track:19QNA6AoLPJvKcf4Oosg2z',
       'spotify:track:2gmdr7WU3kSGRsDxv4tOqA',
       'spotify:track:7vA7BaPdYpdJtt8zVJaqy8',
       'spotify:track:58mtgcQVZ56NgWHKsN94nD']

    track_uris = ','.join(track_ids)
    playlist_data = {
        "name": "Bot Gen"
    }
    res_pl = requests.post(f'https://api.spotify.com/v1/users/21nrpmngofpue4bu35v3b7moq/playlists', json=playlist_data, headers=headers)
    pl_id = res_pl.json()['id']
    print(pl_id)
    print(track_uris)
    track_data = {
        "uris": track_ids
    }
    res_tr = requests.post(f'https://api.spotify.com/v1/playlists/{pl_id}/tracks', json=track_data, headers=headers)
    print(res_tr.json())
    return jsonify(res_tr.json())


@app.route('/cluster_playlist')
def cluster_playlist():
    access_token = request.json['access_token']
    headers = {
        'Authorization': 'Bearer ' + access_token
    }
    track_list = get_tracks(request.json["id"], headers, track_list={})
    track_list = get_track_features(track_list, headers)
    track_list = get_audio_feature(track_list, headers)
    track_data = list(track_list.values())

    track_df = pd.DataFrame(track_data)
    track_df.set_index('id', inplace=True)
    track_df.drop("name", axis=1, inplace=True)
    track_df.drop("artist", axis=1, inplace=True)
    
    tone_X = track_df.drop(["name","artist","explicit","key","mode","time_signature","duration_ms","emotion","timbre"], axis=1)
    tone_model = pickle.load(open('tone.model','rb'))
    track_df['tone'] = tone_model.predict(tone_X)

    hot_tone_X = track_df[track_df['tone'] == 0]
    
    return 'success'



def get_tracks(playlist_id, headers, next_url=None, track_list={}):
    if next_url:
        rtracks = requests.get(next_url, headers=headers)
    else:
        rtracks = requests.get(f'https://api.spotify.com/v1/playlists/{playlist_id}/tracks', headers=headers)
        
    tracks = rtracks.json()
    for track in tracks['items']:
        track_list[track['track']['id']] = {
            'id': track['track']['id'],
            'name': track['track']['name'],
            'duration_ms': track['track']['duration_ms'],
            'popularity': track['track']['popularity'],
            'explicit': track['track']['explicit'],
            'artist': track['track']['artists'][0]['name']
        }

    if tracks['next']:
        return get_tracks(playlist_id, headers, next_url=tracks['next'], track_list=track_list)
    else:
        return track_list


def get_track_features(track_list, headers, iter_index=0):
    id_list = list(track_list.keys())
    start_index = iter_index*100
    end_index = (iter_index+1)*100 if (iter_index+1)*100 < len(id_list) else len(id_list)
    req_id_list = id_list[start_index:end_index]

    rfeatures = requests.get(f'https://api.spotify.com/v1/audio-features/?ids={",".join(req_id_list)}', headers=headers)
    features = rfeatures.json()

    for feature in features['audio_features']:
        track_list[feature['id']]['danceability'] = feature['danceability']
        track_list[feature['id']]['energy'] = feature['energy']
        track_list[feature['id']]['key'] = feature['key']
        track_list[feature['id']]['loudness'] = feature['loudness']
        track_list[feature['id']]['mode'] = feature['mode']
        track_list[feature['id']]['speechiness'] = feature['speechiness']
        track_list[feature['id']]['acousticness'] = feature['acousticness']
        track_list[feature['id']]['instrumentalness'] = feature['instrumentalness']
        track_list[feature['id']]['liveness'] = feature['liveness']
        track_list[feature['id']]['valence'] = feature['valence']
        track_list[feature['id']]['tempo'] = feature['tempo']
        track_list[feature['id']]['time_signature'] = feature['time_signature']

    if end_index != len(id_list):
        iter_index += 1
        return get_track_features(track_list, headers, iter_index=iter_index)
    else:
        return track_list


def get_audio_feature(track_list, headers):
    for track in track_list:
        rfeature = requests.get(f'https://api.spotify.com/v1/audio-analysis/{track["id"]}', headers=headers)
        feature = rfeature.json()
        if 'track' in feature:
            afeature = feature['track']
            segments = feature['segments']

            duration = afeature['duration']
            pitch_values = {'C': 0, 'C#':0, 'D':0, 'D#':0, 'E':0 , 'F':0, 'F#':0, 'G':0, 'G#':0, 'A':0, 'A#':0, 'B':0}
            timbre_values = {'B1': 0, 'B2':0, 'B3':0, 'B4':0, 'B5':0 , 'B6':0, 'B7':0, 'B8':0, 'B9':0, 'B10':0, 'B11':0, 'B12':0}

            for segment in segments:
                seg_duration = segment['duration']
                pitches = segment['pitches']
                pitch_values['C'] += pitches[0] * seg_duration
                pitch_values['C#'] += pitches[1] * seg_duration
                pitch_values['D'] += pitches[2] * seg_duration
                pitch_values['D#'] += pitches[3] * seg_duration
                pitch_values['E'] += pitches[4] * seg_duration
                pitch_values['F'] += pitches[5] * seg_duration
                pitch_values['F#'] += pitches[6] * seg_duration
                pitch_values['G'] += pitches[7] * seg_duration
                pitch_values['G#'] += pitches[8] * seg_duration
                pitch_values['A'] += pitches[9] * seg_duration
                pitch_values['A#'] += pitches[10] * seg_duration
                pitch_values['B'] += pitches[11] * seg_duration

                timbres = segment['timbre']
                timbre_values['B1'] += timbres[0] * seg_duration
                timbre_values['B2'] += timbres[1] * seg_duration
                timbre_values['B3'] += timbres[2] * seg_duration
                timbre_values['B4'] += timbres[3] * seg_duration
                timbre_values['B5'] += timbres[4] * seg_duration
                timbre_values['B6'] += timbres[5] * seg_duration
                timbre_values['B7'] += timbres[6] * seg_duration
                timbre_values['B8'] += timbres[7] * seg_duration
                timbre_values['B9'] += timbres[8] * seg_duration
                timbre_values['B10'] += timbres[9] * seg_duration
                timbre_values['B11'] += timbres[10] * seg_duration
                timbre_values['B12'] += timbres[11] * seg_duration
                

            for note in pitch_values:
                pitch_values[note] = pitch_values[note] / duration

            for basis in timbre_values:
                timbre_values[basis] = timbre_values[basis] / duration

            track['timbre'] = timbre_values
            track['pitches'] = pitch_values

    return track_list


def get_service_token():
    now = datetime.now()
    access_token = None
    token = col_token.find_one({"_id": "service"})
    if token and token['expire_time'] > now:
        access_token = token['access_token']
    else:
        form = {
            'grant_type': 'client_credentials'
        }
        auth_str = f'{client_id}:{client_secret}'
        headers = {
            'Authorization': 'Basic '+str(base64.b64encode(bytes(auth_str,'utf-8')), "utf-8")
        }
        r = requests.post('https://accounts.spotify.com/api/token', data=form, headers=headers)
        if r.status_code == 200:
            access_token = r.json()['access_token']
            if token:
                col_token.update_one({'_id': 'service'}, {'$set': {'access_token': access_token, 'expire_time': now + timedelta(hours=1)}})
            else:
                col_token.insert_one({'_id': 'service', 'access_token': access_token, 'expire_time': now + timedelta(hours=1)})

    return access_token


def get_user_token(user_id):
    now = datetime.now()
    access_token = None
    token = col_token.find_one({"_id": user_id})
    if token and token['expire_time'] > now:
        access_token = token['access_token']
    # else:



if __name__ == '__main__':
   app.run(host='127.0.0.1', port=8888)