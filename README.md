# AI Assistant with Persistent Memory

This AI assistant uses Redis Stack for persistent memory storage, vector similarity search, and JSON document storage. It includes RedisInsight for visual management and monitoring.

## Prerequisites

- Docker and Docker Compose
- Python 3.8+
- Git
- At least 4GB RAM available

## Features

- **Vector Similarity Search** via RediSearch
- **JSON Document Storage** via RedisJSON
- **Visual Management** via RedisInsight
- **Persistent Memory** across restarts
- **Automatic Memory Management**
- **Performance Monitoring**

## Quick Setup

1. **Install Docker and Docker Compose**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install docker.io docker-compose
   
   # Start Docker and enable at boot
   sudo systemctl start docker
   sudo systemctl enable docker
   ```

2. **Add User to Docker Group**
   ```bash
   sudo usermod -aG docker $USER
   # Log out and back in for changes to take effect
   ```

3. **Create Redis Configuration**
   ```bash
   # Create redis.conf in config directory
   cat > config/redis.conf << EOL
   # Memory Management
   maxmemory 2gb
   maxmemory-policy allkeys-lru
   
   # Persistence
   appendonly yes
   appendfsync everysec
   save 60 1
   
   # RediSearch Configuration
   MAXPREFIXEXPANSIONS 100
   MAXAGGREGATERESULTS 100000
   
   # Security (customize these)
   protected-mode yes
   bind 127.0.0.1
   EOL
   ```

4. **Start Redis Stack**
   ```bash
   # Start the services
   docker-compose -f config/docker-compose.yaml up -d
   
   # Verify services are running
   docker ps
   ```

5. **Access RedisInsight**
   - Open http://localhost:8001 in your browser
   - Add a new Redis database:
     - Host: localhost
     - Port: 6380
     - Name: AI-Assistant

## Memory Management

1. **Vector Search Index**
   ```bash
   # Monitor search index
   docker exec -it $(docker ps | grep redis-stack | awk '{print $1}') redis-cli FT.INFO memory_idx
   ```

2. **Memory Usage**
   ```bash
   # Check memory stats
   docker exec -it $(docker ps | grep redis-stack | awk '{print $1}') redis-cli INFO memory
   ```

3. **Backup Data**
   ```bash
   # Manual backup
   docker exec -it $(docker ps | grep redis-stack | awk '{print $1}') redis-cli SAVE
   
   # Copy backup file
   docker cp $(docker ps | grep redis-stack | awk '{print $1}'):/data/dump.rdb ./backup/
   ```

## Monitoring & Maintenance

1. **RedisInsight Features**
   - Real-time memory analysis
   - Slow query monitoring
   - Index management
   - Data browser

2. **Performance Tuning**
   ```bash
   # Check slow log
   docker exec -it $(docker ps | grep redis-stack | awk '{print $1}') redis-cli SLOWLOG GET 10
   
   # Clear slow log
   docker exec -it $(docker ps | grep redis-stack | awk '{print $1}') redis-cli SLOWLOG RESET
   ```

## Troubleshooting

1. **Memory Issues**
   ```bash
   # Check memory usage
   docker stats $(docker ps | grep redis-stack | awk '{print $1}')
   
   # Clear specific keys if needed
   docker exec -it $(docker ps | grep redis-stack | awk '{print $1}') redis-cli KEYS "pattern:*" | xargs redis-cli DEL
   ```

2. **Index Problems**
   ```bash
   # Rebuild search index
   docker exec -it $(docker ps | grep redis-stack | awk '{print $1}') redis-cli FT.DROPINDEX memory_idx
   # The index will be automatically recreated on next access
   ```

3. **Connection Issues**
   ```bash
   # Test Redis connection
   docker exec -it $(docker ps | grep redis-stack | awk '{print $1}') redis-cli ping
   
   # Check logs
   docker logs $(docker ps | grep redis-stack | awk '{print $1}')
   ```

## Security Considerations

1. **Network Security**
   - Redis Stack runs on port 6380 (Redis) and 8001 (RedisInsight)
   - Configure firewall to restrict access:
     ```bash
     sudo ufw allow from 127.0.0.1 to any port 6380
     sudo ufw allow from 127.0.0.1 to any port 8001
     ```

2. **Data Security**
   - All data is persisted in the `redis_data` volume
   - Regular backups are recommended
   - Consider enabling Redis AUTH for production

## Additional Resources

- [Redis Stack Documentation](https://redis.io/docs/stack)
- [RediSearch Guide](https://redis.io/docs/stack/search)
- [RedisJSON Documentation](https://redis.io/docs/stack/json)
- [RedisInsight Documentation](https://redis.io/docs/stack/insight) 