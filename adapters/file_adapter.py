class FileAdapter:
    @staticmethod
    def save_text(filepath: str, text: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)

    @staticmethod
    def read_text(filepath: str) -> str:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read() 