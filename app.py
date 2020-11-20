import pickle
import math
import requests
import base64
import pandas as pd
from kmodes.kmodes import KModes

from flask import Flask, render_template, make_response, redirect, url_for, request, session, jsonify
from urllib.parse import urlencode
from config import DefaultConfig
from datetime import datetime, timedelta
from flask_cors import CORS
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from datetime import datetime

CONFIG = DefaultConfig()

app = Flask(__name__)
CORS(app)

app.secret_key = b'2aa45112fa0e4d38befc0cbde25006e1'
client_id = 'ca83153c363a4fbc8fc683647b84831d'
client_secret = '2aa45112fa0e4d38befc0cbde25006e1'
redirect_uri = 'https://playlist-clustering.herokuapp.com'

letters = 'abcdefghijklmnopqrstuvwxyz0123456789'
stateKey = 'spotify_auth_state'

season_playlist_name = {
    ('summer',): 'Summer',
    ('spring',): 'Spring',
    ('autumn',): 'Autumn',
    ('winter',): 'Winter',
    ('rainy',): 'Rainy',
    ('spring','summer'): 'Sunny',
    ('autumn','summer'): 'Vividly',
    ('summer','winter'): 'Melty',
    ('rainy','summer'): 'Sultry',
    ('autumn','spring'): 'Peaceful',
    ('spring','winter'): 'Snowy',
    ('rainy','spring'): 'Cloudy',
    ('autumn','winter'): 'Windy',
    ('autumn','rainy'): 'Dampy',
    ('rainy','winter'): 'Freezy'
}

emotion_playlist_name = {
    ('happy',): 'Happiness',
    ('chill',): 'Chill',
    ('party',): 'Party',
    ('sad',): 'Sadness',
    ('chill','happy'): 'Comfort',
    ('happy','party'): 'Fun',
    ('happy','sad'): 'Confusion',
    ('chill','party'): 'Rhythm',
    ('party','sad'): 'Conflict',
    ('chill','sad'): 'Melancholy'
}

@app.route('/login', methods=['POST'])
def login():
    access_token = request.json['access_token']
    headers = {
        'Authorization': 'Bearer ' + access_token
    }
    rme = requests.get('https://api.spotify.com/v1/me', headers=headers)
    user = rme.json()
    # rplaylists = requests.get('https://api.spotify.com/v1/me/playlists?limit=50', headers=headers)
    # playlists = rplaylists.json()['items']
    # playlist_items = []
    # print(playlists)
    # for playlist in playlists:
    #     playlist_items.append({
    #         'id': playlist["id"],
    #         'name': playlist["name"],
    #         'images': playlist["images"],
    #         'track_counts': playlist['tracks']['total']
    #     })

    user['playlists'] = get_user_playlist(headers, playlist_items=[])

    return jsonify(user)


@app.route('/create_playlist', methods=['POST'])
def create_playlist():
    access_token = request.json['token']
    headers = {
        'Authorization': 'Bearer ' + access_token
    }
    track_ids = request.json["track_ids"]
    track_ids = ['spotify:track:' + track_id for track_id in track_ids]

    track_uris = ','.join(track_ids)
    playlist_data = {
        "name": request.json["name"]
    }
    res_pl = requests.post(f'https://api.spotify.com/v1/users/{request.json["user_id"]}/playlists', json=playlist_data, headers=headers)
    pl_id = res_pl.json()['id']
    track_data = {
        "uris": track_ids
    }
    res_tr = requests.post(f'https://api.spotify.com/v1/playlists/{pl_id}/tracks', json=track_data, headers=headers)

    return jsonify(res_tr.json())


@app.route('/cluster_playlist', methods=['POST'])
def cluster_playlist():
    print('1',datetime.now())
    access_token = request.json['token']
    headers = {
        'Authorization': 'Bearer ' + access_token
    }
    track_list = get_tracks(request.json["id"], headers, track_list={})
    track_list = get_track_data(track_list, headers)
    track_list = get_track_features(track_list, headers)
    # track_list = get_audio_feature(track_list, headers)
    track_data = list(track_list.values())

    track_df_master = pd.DataFrame(track_data)
    track_df_master.set_index('id', inplace=True)

    track_df = track_df_master.copy()
    track_df.drop("name", axis=1, inplace=True)
    track_df.drop("artist", axis=1, inplace=True)
    track_df.drop("explicit", axis=1, inplace=True)
    track_df.drop("duration_ms", axis=1, inplace=True)
    track_df.drop("time_signature", axis=1, inplace=True)
    track_df.drop("mode", axis=1, inplace=True)
    track_df.drop("key", axis=1, inplace=True)
    track_df['release_date'] = pd.to_numeric(track_df['release_date'].str.split('-',expand=True)[1])
    track_df["release_date"].fillna(track_df["release_date"].median(skipna=True), inplace=True)
    
    print('2',datetime.now())
    tone_model = pickle.load(open('tone.model','rb'))
    tone_X = track_df.copy()
    tone = tone_model.predict(tone_X)
    track_df['tone'] = tone
    hot_track = track_df[track_df['tone'] == 0]
    cold_track = track_df[track_df['tone'] == 1]

    hot_track_X = hot_track.drop(['tone'], axis=1)
    hot_model = pickle.load(open('spring_summer.model','rb'))
    season = hot_model.predict(hot_track_X)
    hot_track_X['season'] = season
    hot_track_X['season'] = hot_track_X['season'].replace(0, 'spring')
    hot_track_X['season'] = hot_track_X['season'].replace(1, 'summer')

    cold_track_X = cold_track.drop(['tone'], axis=1)
    cold_model = pickle.load(open('winter_autumn.model','rb'))
    season = cold_model.predict(cold_track_X)
    cold_track_X['season'] = season
    cold_track_X['season'] = cold_track_X['season'].replace(0, 'autumn')
    cold_track_X['season'] = cold_track_X['season'].replace(1, 'rainy')
    cold_track_X['season'] = cold_track_X['season'].replace(2, 'winter')

    season_sr = cold_track_X['season'].append(hot_track_X['season'])
    track_df = pd.merge(track_df, season_sr, left_index=True, right_index=True, how='left')
    print('3',datetime.now())

    emotion_model = pickle.load(open('emotion.model','rb'))
    emotion_X = track_df.drop(['tone', 'season'], axis=1)
    emotion = emotion_model.predict(emotion_X)
    track_df['emotion'] = emotion
    track_df['emotion'] = track_df['emotion'].replace(0, 'chill')
    track_df['emotion'] = track_df['emotion'].replace(1, 'happy')
    track_df['emotion'] = track_df['emotion'].replace(2, 'party')
    track_df['emotion'] = track_df['emotion'].replace(3, 'sad')
    track_df['emotion'].value_counts()
    print('4',datetime.now())
    for n in range(15,1,-1):
        km_cao = KModes(n_clusters=n, init = "Huang", n_init=15)
        clusters = km_cao.fit_predict(track_df[['emotion','season']])
        min_track_count = pd.Series(clusters).value_counts().min()
        if min_track_count > 9:
            break
    
    track_df['cluster'] = clusters

    print('5',datetime.now())
    clustered_playlist = {}
    for i in range(n):
        cluster = track_df[track_df['cluster']==i].copy()
        cluster['id'] = cluster.index
        cluster = pd.merge(cluster, track_df_master[['name','artist']], left_index=True, right_index=True, how='left')
        
        season_char = None
        emotion_char = None
        season_rank = cluster['season'].value_counts().sort_values(ascending=False)
        emotion_rank = cluster['emotion'].value_counts().sort_values(ascending=False)
        size = cluster.shape[0]
        season_char = (season_rank.index[0],)
        if season_rank.size > 1 and (season_rank[1]/size) > (season_rank[0]/size/2):
            season_char = (season_rank.index[0], season_rank.index[1])

        if emotion_rank.index[0] in ['happy', 'party']:
            if 'sad' in emotion_rank.index:
                if emotion_rank['sad'] > 9:
                    clustered_playlist['s'+i] = {
                        'name': season_playlist_name[tuple(sorted(season_char))] + " " + emotion_playlist_name[tuple(sorted(('sad')))],
                        'tracks': cluster[cluster['emotion'] == 'sad'][['name','artist','season','emotion','id']].to_dict(orient='records')
                    }
                cluster = cluster.drop(cluster[cluster['emotion'] == 'sad'].index)
        elif emotion_rank.index[0] == 'sad':
            if 'party' in emotion_rank.index:
                if emotion_rank['party'] > 9:
                    clustered_playlist['p'+i] = {
                        'name': season_playlist_name[tuple(sorted(season_char))] + " " + emotion_playlist_name[tuple(sorted(('party')))],
                        'tracks': cluster[cluster['emotion'] == 'party'][['name','artist','season','emotion','id']].to_dict(orient='records')
                    }
                cluster = cluster.drop(cluster[cluster['emotion'] == 'party'].index)
            if 'happy' in emotion_rank.index:
                if emotion_rank['happy'] > 9:
                    clustered_playlist['h'+i] = {
                        'name': season_playlist_name[tuple(sorted(season_char))] + " " + emotion_playlist_name[tuple(sorted(('happy')))],
                        'tracks': cluster[cluster['emotion'] == 'happy'][['name','artist','season','emotion','id']].to_dict(orient='records')
                    }
                cluster = cluster.drop(cluster[cluster['emotion'] == 'happy'].index)

        emotion_char = (emotion_rank.index[0],)
        if emotion_rank.size > 1 and (emotion_rank[1]/size) > (emotion_rank[0]/size/2):
            emotion_char = (emotion_rank.index[0], emotion_rank.index[1])

        clustered_playlist[i] = {
            'name': season_playlist_name[tuple(sorted(season_char))] + " " + emotion_playlist_name[tuple(sorted(emotion_char))],
            'tracks': cluster[['name','artist','season','emotion','id']].to_dict(orient='records'),
            'track_counts': size
        }

    print('6',datetime.now())

    return jsonify({'playlists': list(clustered_playlist.values())})


def get_user_playlist(headers, playlist_items=[], next_url=None):
    if next_url:
        rplaylists = requests.get(next_url, headers=headers)
    else:
        rplaylists = requests.get('https://api.spotify.com/v1/me/playlists', headers=headers)
    playlists = rplaylists.json()['items']
    for playlist in playlists:
        playlist_items.append({
            'id': playlist["id"],
            'name': playlist["name"],
            'images': playlist["images"],
            'track_counts': playlist['tracks']['total']
        })

    if rplaylists.json()['next']:
        return get_user_playlist(headers, next_url=rplaylists.json()['next'], playlist_items=playlist_items)
    else:
        return playlist_items


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


def get_track_data(track_list, headers, iter_index=0):
    id_list = list(track_list.keys())
    start_index = iter_index*50
    end_index = (iter_index+1)*50 if (iter_index+1)*50 < len(id_list) else len(id_list)
    req_id_list = id_list[start_index:end_index]

    rdata = requests.get(f'https://api.spotify.com/v1/tracks/?ids={",".join(req_id_list)}', headers=headers)
    data = rdata.json()
    for track in data['tracks']:
        track_list[track['id']]['release_date'] = track['album']['release_date']

    if end_index != len(id_list):
        iter_index += 1
        return get_track_data(track_list, headers, iter_index=iter_index)
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
    for track_id in track_list:
        track = track_list[track_id]
        rfeature = requests.get(f'https://api.spotify.com/v1/audio-analysis/{track_id}', headers=headers)
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
   app.run()
#    app.run(host='127.0.0.1', port=8888)