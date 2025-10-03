# src/data_collection.py
from androguard.core.apk import APK

def get_apk_permissions(apk_path):
    """Extracts declared permissions from an APK file."""
    try:
        apk = APK(apk_path)
        permissions = apk.get_permissions()
        print(f"Successfully extracted {len(permissions)} permissions from {apk_path}")
        return permissions
    except Exception as e:
        print(f"Error processing {apk_path}: {e}")
        return []

if __name__ == '__main__':
    # You would place your APK file in 'data/raw/'
    # Example usage:
    # permissions = get_apk_permissions('data/raw/sample_app.apk')
    # print(permissions)
    pass