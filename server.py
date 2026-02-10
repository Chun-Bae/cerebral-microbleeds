from flask import Flask, send_from_directory, render_template_string, abort, request
import os
import datetime

app = Flask(__name__)

# 결과 폴더 절대 경로 (현재 파일 기준 'results')
# 만약 main.py와 동일한 위치에 이 파일이 있다면
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def get_file_info(path):
    """파일 정보 반환 (크기, 수정일 등)"""
    try:
        stat = os.stat(path)
        size_str = (
            f"{stat.st_size / 1024:.1f} KB"
            if stat.st_size < 1024 * 1024
            else f"{stat.st_size / (1024 * 1024):.1f} MB"
        )
        mtime = datetime.datetime.fromtimestamp(stat.st_mtime).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        return size_str, mtime
    except:
        return "-", "-"


@app.route("/")
def index():
    return list_directory("")


@app.route("/<path:subpath>")
def list_directory(subpath):
    # 보안: 상위 경로 접근 방지
    if ".." in subpath:
        abort(403)

    # 요청된 경로의 절대 경로
    abs_path = os.path.join(RESULTS_DIR, subpath)

    # 존재하지 않으면 404
    if not os.path.exists(abs_path):
        return f"Path not found: {subpath}", 404

    # 파일이면 다운로드/보기
    if os.path.isfile(abs_path):
        directory = os.path.dirname(abs_path)
        filename = os.path.basename(abs_path)
        return send_from_directory(directory, filename)

    # 디렉토리면 목록 보여주기
    try:
        items = sorted(os.listdir(abs_path))
    except PermissionError:
        abort(403)

    # 디렉토리와 파일 분리 및 정렬 (디렉토리 먼저)
    dirs = []
    files = []

    for item in items:
        item_path = os.path.join(abs_path, item)
        if os.path.isdir(item_path):
            dirs.append(item)
        else:
            files.append(item)

    # HTML 템플릿
    html = """
    <!doctype html>
    <html lang="ko">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Results Browser - /{{ subpath }}</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; padding: 2rem; max-width: 900px; margin: 0 auto; background-color: #f8f9fa; }
            .container { background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            h1 { border-bottom: 2px solid #eee; padding-bottom: 0.5rem; margin-top: 0; font-size: 1.5rem; color: #333; }
            ul { list-style: none; padding: 0; }
            li { padding: 0.8rem 0; border-bottom: 1px solid #f0f0f0; display: flex; align-items: center; }
            li:last-child { border-bottom: none; }
            a { text-decoration: none; color: #0366d6; font-weight: 500; flex-grow: 1; }
            a:hover { text-decoration: underline; }
            .meta { font-size: 0.85rem; color: #666; width: 150px; text-align: right; }
            .icon { margin-right: 10px; width: 24px; text-align: center; }
            .back-link { display: inline-block; margin-bottom: 1rem; color: #666; font-weight: bold; }
            .path-nav { background: #eee; padding: 0.5rem; border-radius: 4px; margin-bottom: 1rem; font-family: monospace; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📂 Results Browser</h1>
            
            <div class="path-nav">Current Path: /{{ subpath }}</div>
            
            {% if subpath %}
                {% set parent_path = '/'.join(subpath.split('/')[:-1]) %}
                <a href="/{{ parent_path }}" class="back-link">⬅️ .. (Parent Directory)</a>
            {% endif %}
            
            <ul>
                <!-- Directories -->
                {% for d in dirs %}
                    {% set link = subpath + '/' + d if subpath else d %}
                    <li>
                        <span class="icon">📁</span>
                        <a href="/{{ link }}">{{ d }}/</a>
                        <span class="meta">-</span>
                    </li>
                {% endfor %}
                
                <!-- Files -->
                {% for f in files %}
                    {% set link = subpath + '/' + f if subpath else f %}
                    {% set full_path = os.path.join(abs_path, f) %}
                    {% set size, mtime = get_file_info(full_path) %}
                    <li>
                        <span class="icon">📄</span>
                        <a href="/{{ link }}">{{ f }}</a>
                        <span class="meta">{{ size }}<br>{{ mtime }}</span>
                    </li>
                {% endfor %}
            </ul>
        </div>
    </body>
    </html>
    """

    return render_template_string(
        html,
        subpath=subpath,
        dirs=dirs,
        files=files,
        abs_path=abs_path,
        os=os,
        get_file_info=get_file_info,
    )


if __name__ == "__main__":
    print(f"🚀 Starting Results Server...")
    print(f"📂 Serving directory: {RESULTS_DIR}")
    print(f"👉 Access at: http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
