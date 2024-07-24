#!/usr/bin/env python

import sqlite3
import logging

logging.basicConfig(level=logging.INFO)

def clear_cache(db_name='ai_assistant.db'):
    conn = sqlite3.connect(db_name)
    with conn:
        logging.info("Clearing cached responses...")
        conn.execute('DELETE FROM Data')
        logging.info("Cache cleared.")

if __name__ == "__main__":
    clear_cache()
