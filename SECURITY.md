# Security Policy — Media AI Local

## Uso

O projeto foi desenvolvido para execução local e educacional. Não o exponha diretamente à internet sem adicionar autenticação, autorização, rate limiting, observabilidade e controles de armazenamento.

## Conteúdo sensível

Não envie para o sistema documentos, imagens ou informações pessoais de terceiros sem autorização apropriada.

O repositório não deve receber:
- arquivos de modelos;
- bancos locais;
- imagens ou vídeos gerados;
- tokens e chaves;
- documentos pessoais.

## Filtros

O módulo de segurança contém heurísticas de entrada. Elas ajudam a reduzir usos indevidos, mas não constituem uma barreira de segurança completa.

## Vulnerabilidades

Evite publicar segredos ou dados reais em issues públicas. Ao reportar uma vulnerabilidade, descreva o problema sem incluir credenciais ou dados pessoais.
