from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dramatiq.brokers.redis import RedisBroker
from dramatiq

def get_db_connection1():
    DATABASE_URI = 'sqlite:////Users/yash/Documents/Calra AI/clara.db'
    # For other databases, adjust the connection string accordingly
    # PostgreSQL example: 'postgresql://user:password@localhost/your_database'
    engine = create_engine(DATABASE_URI)
    Session = sessionmaker(bind=engine)
    return Session()

DATABASE_URI = 'sqlite:////Users/yash/Documents/Calra AI/clara.db'


engine = create_engine(DATABASE_URI,pool_size=10) # Set pool size to 10
SessionLocal = sessionmaker(autocommit=False,autoflush=False,bind=engine)

def get_idb():
    db=SessionLocal()
    try:
        yield db
    except:
        db.close()

def get_db():
    return next(get_idb())