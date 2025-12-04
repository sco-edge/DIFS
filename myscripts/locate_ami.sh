#!/bin/bash
# Purpose: For trying to locate the Amazon machine image (AMI) that was recommended in my README for General Setup; the ami that was suggested was 'ami-036de08e2e59b4abc'
# [[https://stackoverflow.com/questions/59678721/how-to-find-an-amazon-ec2-ami-if-i-only-have-the-id][How to find an Amazon EC2 AMI if I only have the id?]]

for region in $(aws ec2 describe-regions --query 'Regions[].RegionName' --output text | tr '\t' '\n')
 do 
   aws ec2 describe-images --image-ids ami-036de08e2e59b4abc --region "$region" &>/dev/null;
   if [[ "$?" -eq 0 ]]; 
     then echo "AMI found in region ${region}!" && break
   fi 
done
