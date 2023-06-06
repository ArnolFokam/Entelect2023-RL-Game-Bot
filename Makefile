.PHONY: push pushb pusht pushr pushf pushn

# git commands
push:
	git add .
	git commit -m "$(message)"
	git push

pushb: message=bug fix
pushb: push

pushf: message=feature enhancement
pushf: push

pushr: message=refactoring
pushr: push

pushn: message=update notes
pushn: push

pusht: message=testing
pusht: push