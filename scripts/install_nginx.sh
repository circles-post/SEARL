# 默认参数
UPSTREAM_PORTS=()  # 默认为空数组
LISTEN_PORT=9000   # 默认监听端口

# 解析命令行参数
for arg in "$@"; do
    case "$arg" in
        --upstream=*)
            # 解析逗号分隔的端口列表（例如 --upstream=8000,8001,8002）
            IFS=',' read -ra ports <<< "${arg#*=}"
            for port in "${ports[@]}"; do
                UPSTREAM_PORTS+=("127.0.0.1:$port")
            done
            shift
            ;;
        --port=*)
            LISTEN_PORT="${arg#*=}"
            shift
            ;;
        *)
            # 未知参数
            echo "Unknown parameter: $arg"
            echo "Usage: $0 [--upstream=port1,port2,...] [--port=listen_port]"
            exit 1
            ;;
    esac
done

# 如果没有传入 upstream 参数，使用默认值
if [ ${#UPSTREAM_PORTS[@]} -eq 0 ]; then
    UPSTREAM_PORTS=("127.0.0.1:8000" "127.0.0.1:8001")  # 默认值
fi

# 创建安装目录
export http_proxy=http://172.18.207.111:8420
export https_proxy=http://172.18.207.111:8420

mkdir -p ~/nginx && cd ~/nginx

# 下载 Nginx 源码
wget http://nginx.org/download/nginx-1.25.3.tar.gz
tar -xzf nginx-1.25.3.tar.gz
cd nginx-1.25.3

# 编译安装（指定用户目录）
./configure --prefix=$HOME/nginx --without-http_rewrite_module
make && make install

# 定义变量
NGINX_CONF="$HOME/nginx/conf/nginx.conf"

# 备份原配置文件
cp "$NGINX_CONF" "$NGINX_CONF.bak" 2>/dev/null || echo "Creating new config file"

# 生成新的配置文件
cat > "$NGINX_CONF" <<EOF
worker_processes  1;

events {
    worker_connections  1024;
}

http {
    upstream vllm_servers {
        # 自动生成的服务器列表
        $(for server in "${UPSTREAM_PORTS[@]}"; do echo "        server $server;"; done)
        least_conn;
    }

    server {
        listen       $LISTEN_PORT;
        server_name  localhost;

        location / {
            proxy_pass http://vllm_servers;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            
            # 长连接设置
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            
            # 超时设置（适合LLM长响应）
            proxy_connect_timeout 300s;
            proxy_send_timeout 300s;
            proxy_read_timeout 300s;
            send_timeout 300s;
        }
    }
}
EOF

echo "Nginx configuration updated at $NGINX_CONF"
echo "Upstream servers configured:"
printf " - %s\n" "${UPSTREAM_PORTS[@]}"

~/nginx/sbin/nginx
