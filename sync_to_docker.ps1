$targetContainer = "mettalog"
$hostFolder = "./synthcl/."
$containerFolder = "/home/user/devspace/synthcl"

if (-not $(docker ps --filter "name=$targetContainer" --filter "status=running" -q)) {
    echo "Container $targetContainer has not been started."
    exit 1
}

$folderExists = $(docker exec $targetContainer bash -c "[ -d $containerFolder ] && echo 'exists'")
if ($folderExists -eq "exists") {
    docker exec $targetContainer bash -c "rm -r $containerFolder"
    echo "Deleted $containerFolder from $targetContainer"
}

docker cp $hostFolder "${targetContainer}:${containerFolder}"