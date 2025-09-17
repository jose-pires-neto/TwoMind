# 🧠 Segundo Cérebro - Sistema de Memória Inteligente

Um sistema pessoal de organização de memórias com IA, visualização gráfica e acesso global via Cloudflare Tunnel.

## ✨ Funcionalidades

- 🔍 **Busca Inteligente**: Encontre suas memórias usando linguagem natural
- 🤖 **IA Integrada**: Respostas contextuais baseadas em suas memórias
- 🌐 **Grafo Visual**: Visualize conexões entre suas ideias
- 🔒 **Seguro**: Protegido por senha e criptografia
- 🌍 **Acesso Global**: Use de qualquer lugar via túnel Cloudflare
- 💾 **Dados Locais**: Suas memórias ficam no seu computador

## 🚀 Instalação Rápida

### 1. Baixe os Arquivos
Faça download de todos os arquivos do projeto:
- `app.py`
- `requirements.txt` 
- `segundo_cerebro.html`
- `install.py`

### 2. Execute o Instalador
```bash
python install.py
```

O instalador irá:
- ✅ Verificar compatibilidade do Python
- ✅ Instalar todas as dependências
- ✅ Baixar modelos de IA
- ✅ Criar atalho na área de trabalho

### 3. Inicie a Aplicação
```bash
python app.py
```

### 4. Configure na Primeira Execução
- Digite uma senha forte (mín. 8 caracteres)
- Aguarde o túnel Cloudflare ser criado
- Anote a URL externa fornecida

## 🌐 Como Usar

### Acesso Local
```
http://127.0.0.1:5000
```

### Acesso Global
Após iniciar, você receberá uma URL como:
```
https://abc123.trycloudflare.com
```

### Funcionalidades Principais

1. **Adicionar Memórias**: Clique no botão "+" e digite sua memória
2. **Buscar**: Use o botão de busca para encontrar memórias relacionadas
3. **IA**: Botão "✨" para perguntas respondidas pela IA
4. **Visualizar**: Clique nos nós do grafo para ler memórias completas

## 🔧 Requisitos do Sistema

- **Python**: 3.8 ou superior
- **RAM**: Mínimo 4GB (recomendado 8GB)
- **Espaço**: 2GB livres para modelos de IA
- **Internet**: Para túnel Cloudflare e download inicial

## 🛡️ Segurança

- 🔐 **Autenticação**: Protegido por senha forte
- 🔒 **Sessões**: Expiram automaticamente em 24h
- 🌐 **Túnel Criptografado**: Todo tráfego via HTTPS
- 💾 **Dados Locais**: Nada é enviado para servidores externos

## 📱 Compatibilidade

### Sistemas Operacionais
- ✅ Windows 10/11
- ✅ macOS 10.15+
- ✅ Linux (Ubuntu, Debian, etc.)

### Navegadores
- ✅ Chrome/Edge (recomendado)
- ✅ Firefox
- ✅ Safari
- ✅ Navegadores móveis

## 🐛 Solução de Problemas

### Python não encontrado
```bash
# Windows
python --version

# macOS/Linux  
python3 --version
```

### Erro de dependências
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Túnel Cloudflare não inicia
- Verifique conexão com internet
- Reinicie a aplicação
- Tente aguardar 30 segundos

### Modelos de IA não carregam
- Verifique espaço disponível (2GB+)
- Primeira execução pode demorar 5-10 minutos
- IA é opcional - busca funciona sem ela

## 🎯 Dicas de Uso

1. **Memórias Detalhadas**: Escreva com contexto para melhor busca
2. **Categorize**: Use palavras-chave consistentes
3. **Backup**: Copie periodicamente os arquivos `.faiss` e `.pkl`
4. **Performance**: Feche outros programas pesados durante uso
5. **Mobile**: Interface funciona perfeitamente em celulares

## 📂 Estrutura de Arquivos

```
segundo-cerebro/
├── app.py                 # Servidor principal
├── requirements.txt       # Dependências Python
├── segundo_cerebro.html   # Interface web
├── install.py            # Instalador automático
├── my_index.faiss        # Índice de busca (criado automaticamente)
├── notes.pkl             # Suas memórias (criado automaticamente)
├── user_config.json      # Configurações (criado automaticamente)
└── cloudflared/          # Túnel Cloudflare (baixado automaticamente)
```

## ⚡ Performance

- **Busca**: < 1 segundo para milhares de memórias
- **IA**: 2-5 segundos para respostas
- **Sync**: Conexões calculadas automaticamente
- **Grafo**: Suporta centenas de nós fluidamente

## 🔄 Atualizações

Para atualizar, substitua apenas o `app.py` - seus dados são preservados.

## 📞 Suporte

Em caso de problemas:
1. Verifique os requisitos do sistema
2. Execute novamente o `install.py`
3. Reinicie a aplicação
4. Verifique logs no terminal

---

**🧠 Segundo Cérebro** - Transforme seu computador em um sistema inteligente de memórias!