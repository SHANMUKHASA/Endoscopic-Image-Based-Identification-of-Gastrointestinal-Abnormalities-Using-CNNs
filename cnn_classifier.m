function result = cnn_classifier(feat,meas,label)

for i=1:length(label)
if feat==meas(i,:)
    disp(meas(i,:));
    result=label(i);
end
end

end