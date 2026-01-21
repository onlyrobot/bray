set -o errexit; export SHELLOPTS
export CONDA_DIR=/tmp/pymp-conda; mkdir -p $CONDA_DIR
export CONDA_ENVS_PATH=$CONDA_DIR/conda/envs

if [ ! -d $CONDA_DIR/conda ]; then
if [ $(uname) == "Darwin" ]; then CONDA_NAME=MacOSX-arm64
else CONDA_NAME=Linux-x86_64; fi
if [ ! -f Miniconda3-latest-$CONDA_NAME.sh ]; then
CONDA_URL=https://repo.anaconda.com/miniconda
curl -sL $CONDA_URL/Miniconda3-latest-$CONDA_NAME.sh -O; fi
echo "安装 Conda 到 $CONDA_DIR/conda ..."
bash -i Miniconda3-latest-$CONDA_NAME.sh -b -p $CONDA_DIR/conda
cp $PWD/.condarc $CONDA_DIR/conda/ ; fi

CURRENT_CONDA_ENV=$PWD/conda/$DIST_CONDA_ENV
CACHED_CONDA_ENV=$PWD/cache/$DIST_CONDA_ENV
if [ -n "$DIST_DOCKER_IMAGE" ]; then
CACHED_CONDA_ENV=$CACHED_CONDA_ENV.${DIST_DOCKER_IMAGE////_}; fi
CONDA_ENV_PATH=$CONDA_ENVS_PATH/$DIST_CONDA_ENV

LOCAL_CONDA_ENV=$CONDA_DIR/$DIST_CONDA_ENV
if [[ ! -d $CONDA_ENV_PATH || ! -f $LOCAL_CONDA_ENV ]]; then

if [ -f $CACHED_CONDA_ENV ] && cmp -s "$CURRENT_CONDA_ENV" \
"$CACHED_CONDA_ENV"; then
echo "存在Conda环境缓存 $CACHED_CONDA_ENV 直接使用..."
rm -rf $LOCAL_CONDA_ENV $CONDA_ENV_PATH
tar xf $CACHED_CONDA_ENV.tar -C $CONDA_ENVS_PATH
cp $CACHED_CONDA_ENV $LOCAL_CONDA_ENV

elif [ ! -d $CONDA_ENV_PATH ]; then
echo "创建Conda环境 $DIST_CONDA_ENV 到 $CONDA_ENV_PATH ..."
rm -f $$LOCAL_CONDA_ENV
$CONDA_DIR/conda/bin/conda create -n $DIST_CONDA_ENV -y; fi; fi

echo "激活并安装 Conda 环境 $CONDA_ENV_PATH ..."
source $CONDA_DIR/conda/bin/activate $DIST_CONDA_ENV

if [ ! -f $LOCAL_CONDA_ENV ] || ! cmp -s "$CURRENT_CONDA_ENV" \
"$LOCAL_CONDA_ENV"; then
rm -f $LOCAL_CONDA_ENV; source $CURRENT_CONDA_ENV
cp $CURRENT_CONDA_ENV $LOCAL_CONDA_ENV; else 
echo "Conda 环境已存在且为最新版本，跳过安装依赖"; fi

if [ ! -f $CACHED_CONDA_ENV ] || ! cmp -s "$CACHED_CONDA_ENV" \
"$LOCAL_CONDA_ENV"; then
echo "开始自动缓存Conda环境，加速下次启动..."
tar cf $LOCAL_CONDA_ENV.tar -C $CONDA_ENVS_PATH $DIST_CONDA_ENV
rm -f $CACHED_CONDA_ENV $CACHED_CONDA_ENV.tar
mv $LOCAL_CONDA_ENV.tar $CACHED_CONDA_ENV.tar
cp $LOCAL_CONDA_ENV $CACHED_CONDA_ENV; fi

if [ ! -f $DIST_TRIAL_PATH/env.sh ]; then return; fi
echo "初始化脚本环境..."; source $DIST_TRIAL_PATH/env.sh