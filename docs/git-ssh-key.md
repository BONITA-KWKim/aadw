# GIT ssh 등록

## git ssh key 생성
ssh-keygen -t rsa -C "usarA@email.com" -f "key_file_name"

- 생성 후 key_file_name, key_file_name.pub 파일 생성
- ~/.ssh 폴더 아래로 이동

## config 파일 생성
```
# Personal account-userA 이건 주석이라 아무 내용이나 적어도 상관없습니다. 
Host github.com-userA
  HostName github.com
  User git
  IdentityFile ~/.ssh/key_file_name
```

## key 파일 등록
```
eval $(ssh-agent -s)
ssh-add ~/.ssh/key_file_name
```


## project에 설정파일 생성
```
cd project
cat > .gitconfig
[user]
  email = userA@email.com
```
- 프로젝트 내 git 설정 파일 생성
- 프로젝트의 user 등록


## key 파일 등록 확인
ssh -T github.com-userA


## github setting에 ssh 키 등록
1. github에 접속
2. Setting > SSH key > add ssh key
3. key_file_name.pub 내용을 복사 후 등록


## project 원격 저장소 변경
```
git remote set-url origin git@github.com:${github name}/projectA.git
git remote -v 
```


## project push 확인
git push -u origin main


