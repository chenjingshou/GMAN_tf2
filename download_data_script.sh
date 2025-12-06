#!/bin/bash

# METR-LA数据集下载脚本
# 支持多种下载方式

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================"
echo "  METR-LA数据集下载工具"
echo "======================================${NC}"

# 创建数据目录
DATA_DIR="data/METR-LA"
mkdir -p ${DATA_DIR}

# Google Drive 文件ID（需要从DCRNN仓库获取实际ID）
# 这些是示例ID，请替换为实际的文件ID
METR_LA_H5_ID="10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX"
ADJ_MX_PKL_ID="your_adj_mx_file_id"

echo -e "\n${YELLOW}请选择下载方式:${NC}"
echo "1. 使用gdown从Google Drive下载（推荐，需要网络访问Google）"
echo "2. 从DCRNN仓库克隆并获取数据链接"
echo "3. 手动下载指导"
echo "4. 使用wget下载（如果有直接链接）"
read -p "请输入选项 (1-4): " choice

case $choice in
    1)
        echo -e "\n${GREEN}使用gdown下载...${NC}"
        
        # 检查gdown是否安装
        if ! command -v gdown &> /dev/null; then
            echo "安装gdown..."
            pip install gdown
        fi
        
        echo -e "\n${YELLOW}注意: 由于DCRNN数据集在Google Drive上，请访问以下链接获取数据:${NC}"
        echo "https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX"
        echo ""
        echo "或者使用以下命令手动下载："
        echo "gdown --folder https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX -O ${DATA_DIR}/"
        
        read -p "是否继续尝试自动下载? (y/n) " auto_download
        if [[ $auto_download =~ ^[Yy]$ ]]; then
            cd ${DATA_DIR}
            gdown --folder https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX
            cd ../..
        fi
        ;;
        
    2)
        echo -e "\n${GREEN}从DCRNN仓库获取...${NC}"
        
        # 克隆DCRNN仓库到临时目录
        TEMP_DIR="/tmp/DCRNN_temp"
        if [ -d "$TEMP_DIR" ]; then
            echo "删除旧的临时目录..."
            rm -rf $TEMP_DIR
        fi
        
        echo "克隆DCRNN仓库..."
        git clone https://github.com/liyaguang/DCRNN.git $TEMP_DIR
        
        echo -e "\n${YELLOW}请按照以下步骤操作:${NC}"
        echo "1. 访问 DCRNN README 中的数据链接:"
        echo "   Google Drive: https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX"
        echo "   百度云: 见 ${TEMP_DIR}/README.md"
        echo ""
        echo "2. 下载以下文件:"
        echo "   - metr-la.h5 (交通流量数据)"
        echo "   - adj_mx.pkl (邻接矩阵，可选)"
        echo ""
        echo "3. 将下载的文件放入: ${DATA_DIR}/"
        
        read -p "按Enter键继续..." dummy
        ;;
        
    3)
        echo -e "\n${GREEN}手动下载指导${NC}"
        echo -e "${YELLOW}请按照以下步骤手动下载数据:${NC}"
        echo ""
        echo "步骤1: 访问数据源"
        echo "  Google Drive: https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX"
        echo "  百度云: 在DCRNN仓库README中查找链接"
        echo ""
        echo "步骤2: 下载文件"
        echo "  必需文件:"
        echo "    - metr-la.h5 (~15MB, 交通流量数据)"
        echo "  可选文件:"
        echo "    - adj_mx.pkl (~5MB, 邻接矩阵)"
        echo "    - distances_la_2012.csv (传感器距离)"
        echo "    - graph_sensor_ids.txt (传感器ID列表)"
        echo ""
        echo "步骤3: 放置文件"
        echo "  将下载的文件放入: $(pwd)/${DATA_DIR}/"
        echo ""
        echo "步骤4: 验证下载"
        echo "  运行: python test_environment.py"
        ;;
        
    4)
        echo -e "\n${GREEN}使用wget下载${NC}"
        echo -e "${YELLOW}注意: 此选项需要直接下载链接${NC}"
        echo ""
        read -p "是否有metr-la.h5的直接下载链接? (y/n) " has_link
        
        if [[ $has_link =~ ^[Yy]$ ]]; then
            read -p "请输入metr-la.h5的下载链接: " h5_url
            echo "下载 metr-la.h5..."
            wget -O ${DATA_DIR}/metr-la.h5 "$h5_url"
            
            read -p "是否下载adj_mx.pkl? (y/n) " download_adj
            if [[ $download_adj =~ ^[Yy]$ ]]; then
                read -p "请输入adj_mx.pkl的下载链接: " adj_url
                echo "下载 adj_mx.pkl..."
                wget -O ${DATA_DIR}/adj_mx.pkl "$adj_url"
            fi
        else
            echo -e "${RED}没有直接链接，请使用其他方式下载${NC}"
        fi
        ;;
        
    *)
        echo -e "${RED}无效选项${NC}"
        exit 1
        ;;
esac

# 验证下载
echo -e "\n${BLUE}======================================"
echo "  验证数据文件"
echo "======================================${NC}"

check_file() {
    local file=$1
    local desc=$2
    
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        echo -e "${GREEN}✓${NC} $desc: $file ($size)"
        return 0
    else
        echo -e "${RED}✗${NC} $desc: $file (未找到)"
        return 1
    fi
}

check_file "${DATA_DIR}/metr-la.h5" "交通流量数据"
has_h5=$?

check_file "${DATA_DIR}/adj_mx.pkl" "邻接矩阵"
has_adj=$?

echo ""
if [ $has_h5 -eq 0 ]; then
    echo -e "${GREEN}✓ 必需文件已准备好！${NC}"
    
    if [ $has_adj -eq 0 ]; then
        echo -e "${GREEN}✓ 可选文件也已下载！${NC}"
    else
        echo -e "${YELLOW}! 邻接矩阵未下载，训练时会自动生成${NC}"
    fi
    
    echo -e "\n${GREEN}下一步:${NC}"
    echo "1. 运行环境测试: python test_environment.py"
    echo "2. 预处理数据: python preprocess_data.py"
    echo "3. 开始训练: python train.py --config config.yaml"
else
    echo -e "${RED}✗ 缺少必需文件，请重新下载${NC}"
    echo ""
    echo -e "${YELLOW}帮助信息:${NC}"
    echo "如果下载失败，可以尝试:"
    echo "1. 使用浏览器访问Google Drive链接手动下载"
    echo "2. 使用代理或VPN访问Google服务"
    echo "3. 从百度云下载（链接在DCRNN的README中）"
    echo "4. 询问已有数据的同学或导师"
fi

echo ""
echo -e "${BLUE}======================================"
echo "  数据下载工具结束"
echo "======================================${NC}"
