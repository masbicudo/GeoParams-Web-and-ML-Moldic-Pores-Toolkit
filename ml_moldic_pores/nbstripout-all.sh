#!/bin/bash

dkgray=[90m;red=[91m;green=[92m;yellow=[93m;blue=[94m;magenta=[95m
cyan=[96m;white=[97m;black=[30m;dkred=[31m;dkgreen=[32m;dkyellow=[33m
dkblue=[34m;dkmagenta=[35m;dkcyan=[36m;gray=[37m;cdef=[0m

echo nbstripout-all v1.3

_RESTORE=
_FORCE=
_DIR=
_DISCARD=
while [[ $# -gt 0 ]]
do
  key="$1"
  case $key in
    -r|--restore)      _RESTORE=Y         ;;
    -f|--force)        _FORCE=Y           ;;
    -rf)               _RESTORE=Y;_FORCE=Y;;
    -d)                _DISCARD=Y         ;;
    -h|--help)         _HELP=Y            ;;
    *)                 _DIR="$key"        ;;
  esac
  shift
done

if [ ! -z "$_HELP" ]
then
  col12="[50D""[12C"
  echo $blue$(basename -- $0) $cdef[$cyan"dir_name"$cdef] [$cyan"options"$cdef]
  echo $cyan"dir_name"$cdef $col12                       "Base directory"
  echo $cyan"options"$cdef":"
  echo $blue"["$yellow"none"$blue"]"$cdef $col12         "Default action: asks to restore backup if any, otherwise run nbstripout"
  echo $yellow"-r"$blue"|"$yellow"--restore"$cdef $col12 "Restore backup files"
  echo $yellow"-f"$blue"|"$yellow"--force"$cdef $col12   "If "$yellow"-r"$cdef" then force backup restoration, even if backup is older"
  echo $yellow"-rf"$cdef $col12                          "Same as "$yellow"-r"$cdef" "$yellow"-f"$cdef". Restore backup files and force backup restoration, even if backup is older"
  echo $yellow"-h"$blue"|"$yellow"--help"$cdef $col12    "Show this help screen"
  exit 0
fi

if ! which nbstripout >/dev/null 2>&1
then
    echo $red"nbstripout is not installed"$cdef
    exit 1
fi

if [ ! -z "$_DIR" ]
then
  pushd "$_DIR" >/dev/null 2>&1 || exit 1
fi

# https://unix.stackexchange.com/questions/9496/looping-through-files-with-spaces-in-the-names
# https://www.cyberciti.biz/tips/handling-filenames-with-spaces-in-bash.html
OIFS="$IFS"
IFS=$'\n'
fileArray=($(find . -type f -name '*.ipynb.bkp'))
IFS="$OIFS"
_OPTION=
for (( i=0; i<${#fileArray[@]}; i++ ));
do
  bkp_filename="${fileArray[$i]}"
  while ! [[ ${_OPTION,,} =~ ^[yndf]$ ]];
  do
    echo $white"Would you like to restore backup files?"$cdef
    echo $blue"["$yellow"Y"$blue"]"$white"es"$cdef" - just restore if original file was not changed after backup"
    echo $blue"["$yellow"N"$blue"]"$white"o"$cdef" - run nbstripout and replace backups"
    echo $white"Other options to do with backups:"$cdef
    echo $blue"["$yellow"D"$blue"]"$white"iscard"$cdef" - delete backups and nothing else"
    echo $blue"["$yellow"F"$blue"]"$white"orce"$cdef" - force restore even if original file was changed after backup"
    printf "Chose: "$yellow
    read _OPTION >/dev/null
    printf $cdef
    if [[ "${_OPTION,,}" == "f" ]]
    then
      _RESTORE=Y
      _FORCE=Y
    elif [[ "${_OPTION,,}" == "y" ]]
    then
      _RESTORE=Y
    elif [[ "${_OPTION,,}" == "d" ]]
    then
      _DISCARD=Y
    fi
  done
  
  # option is N we get out of the restoration loop
  if [ "$_DISCARD" != "Y" ] && [ "$_RESTORE" != "Y" ]; then
    break
  fi
  
  # options F, Y and D
  orig_filename=$(echo "$bkp_filename" | sed -r "s/.bkp$//")
  if [ "$_RESTORE" = "Y" ]; then
    if [ "$_FORCE" = "Y" ] || [ ! -f "$orig_filename" ] || [ ! "$bkp_filename" -ot "$orig_filename" ]; then
      echo $green"Restoring"$cdef $white"'$bkp_filename'"$cdef
      cp "$bkp_filename" "$orig_filename"
      rm -rf "$bkp_filename"
    else
      echo $red"Cannot restore"$cdef $white"'$bkp_filename' (file was changed after backup)"$cdef
    fi
  else # [ "$_DISCARD" = "Y" ]
    echo $red"Deleting"$cdef $white"'$bkp_filename'"$cdef
    rm -rf "$bkp_filename"
  fi
done

# if restoring or discarding backups, we are finished by now
if [ "$_RESTORE" = "Y" ]; then exit 0; fi
if [ "$_DISCARD" = "Y" ]; then exit 0; fi

# otherwise, we are ready to strip out ipynb files
OIFS="$IFS"
IFS=$'\n'
fileArray=($(find . -type f -name '*.ipynb'))
IFS="$OIFS"
for (( i=0; i<${#fileArray[@]}; i++ ));
do
  filename="${fileArray[$i]}"
  case "$filename" in 
    *.ipynb_checkpoints*)
      # skip files inside the `.ipynb_checkpoints` folder
      ;;
    *)
      echo $yellow"Striping out"$cdef $white"'$filename'"$cdef
      cp -a "$filename" "$filename.bkp"
      nbstripout "$filename"
      touch -r "$filename.bkp" "$filename"
      ;;
  esac
done
echo -e $blue"\nCommit your work now, and then run "$(basename -- $0)" again to restore files"$cdef

if [ ! -z "$_DIR" ]
then
  popd >/dev/null 2>&1 || exit 1
fi

read -p "Press enter" _
