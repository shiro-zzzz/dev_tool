# 1. 全局开关参数（必须在最上面）
ARG BUILD_ENV=dev

# ==========================================
# 2. 【核心点】真正的“基础镜像”在这里定义！
# 命名为 base，把所有环境都需要的公共操作写在这里
# ==========================================
FROM node:18-alpine AS base
WORKDIR /app
# 比如所有环境都需要先安装依赖，就在 base 里做，避免后面的分支重复执行
COPY package.json package-lock.json ./
RUN npm install

# ==========================================
# 3. 动态配置分支 A：开发环境 (继承自 base)
# ==========================================
FROM base AS config-dev
ARG API_URL="https://api-dev.mycompany.com"
ARG LOG_LEVEL="debug"
# 将 ARG 转存为 ENV，确保最终阶段和运行时可用
ENV VITE_API_URL=${API_URL} \
    VITE_LOG_LEVEL=${LOG_LEVEL}

# ==========================================
# 4. 动态配置分支 B：生产环境 (继承自 base)
# ==========================================
FROM base AS config-prod
ARG API_URL="https://api.mycompany.com"
ARG LOG_LEVEL="error"
ENV VITE_API_URL=${API_URL} \
    VITE_LOG_LEVEL=${LOG_LEVEL}

# ==========================================
# 5. 最终阶段：动态选择分支
# ==========================================
# 这里根据外部传入的 BUILD_ENV（dev 或 prod）来决定以哪个 config 分支为基础
FROM config-${BUILD_ENV} AS final

# 把剩下的业务代码拷贝进来
COPY . .

# 此时，无论选择哪个分支，基础环境（node:18 + 已安装的依赖）和当前环境的专属 ENV 都已经就绪了
RUN echo "当前构建指向的 API 是: ${VITE_API_URL}"

# 执行最终的启动命令
CMD ["npm", "run", "start"]