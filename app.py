
import streamlit as st
import cv2
import tempfile
import mediapipe as mp
from PIL import Image
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="IA Soccer – Analyse Biomécanique", layout="wide")
st.title("🎥 IA Soccer – Analyse Biomécanique du Passe")

video_file = st.file_uploader("📤 Importer une courte vidéo du passe (mp4 recommandé)", type=["mp4", "mov", "avi"])

if video_file is not None:
    st.video(video_file)
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    last_frame = None
    points_detectés = False

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            if results.pose_landmarks:
                points_detectés = True
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
            last_frame = image
        cap.release()

    if points_detectés and last_frame is not None:
        st.success("✅ Détection du squelette réussie.")
        image_pil = Image.fromarray(last_frame)
        st.image(image_pil, caption="🦴 Image du joueur avec squelette détecté", use_column_width=True)

        st.markdown("### 🧠 Analyse IA – Posture lors du passe")
        prompt = """
Un joueur effectue un passe dans une vidéo. Sur l'image capturée, on voit le squelette avec les points du corps.

Fais une analyse technique du geste :
- Position du tronc
- Pied d’appui
- Surface de contact du pied
- Posture des bras
- Équilibre général

Puis propose 3 conseils précis pour améliorer l'efficacité du passe.

Réponds comme un entraîneur professionnel de football jeunesse.
"""
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Tu es un entraîneur technique de football spécialisé en biomécanique."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            st.markdown(response.choices[0].message.content)
        except Exception as e:
            st.error(f"❌ Erreur IA : {e}")
    else:
        st.warning("Aucun squelette n’a été détecté. Essayez avec une autre vidéo plus claire ou plus proche.")
