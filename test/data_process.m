function data_process
    clear;clc;clf;
    dir0 = 'test_mycnn/20170229_test200/';
%     calculate_mean_error(dir0)
    for_show(dir0, 5)
end

function mean_error = calculate_mean_error(dir_test)
    
    mean_error = 0;
    for i = 0 : 199
        keyWord = num2str(i);
        label = importdata([dir_test 'label_' keyWord '.txt']);
        predict = importdata([dir_test 'predict_' keyWord '.txt']);

        result = 0;
        for j = 1 : 2 : 7
            result = result + sqrt((label(j)-predict(j))^2 + (label(j+1)-predict(j+1))^2);
        end
        mean_error = mean_error + result / 4;
    end
    mean_error = mean_error / 200;
end

function result2 = for_show(dir_test, i)
    keyWord = num2str(i);
    img1 = imread([dir_test 'image_' keyWord '_1.jpg']);
    img2 = imread([dir_test 'image_' keyWord '_2.jpg']);
    h1 = importdata([dir_test 'h1_' keyWord '.txt']);
    label = importdata([dir_test 'label_' keyWord '.txt']);
    predict = importdata([dir_test 'predict_' keyWord '.txt']);
    h2_real = h1 + label;
    h2_predict = h1 + predict;

    figure(1);
    set(gcf,'unit','centimeters','position',[1 2 20 12.36]);
    set(gca,'position',[.1 .1 0.8 .8]);
    subplot(1,2,1);
    imshow(img1);
    line([h1(2) h1(4)],[h1(1) h1(3)]);
    line([h1(4) h1(6)],[h1(3) h1(5)]);
    line([h1(6) h1(8)],[h1(5) h1(7)]);
    line([h1(8) h1(2)],[h1(7) h1(1)]);

    subplot(1,2,2);
    imshow(img2);
    line([h2_real(2) h2_real(4)],[h2_real(1) h2_real(3)]);
    line([h2_real(4) h2_real(6)],[h2_real(3) h2_real(5)]);
    line([h2_real(6) h2_real(8)],[h2_real(5) h2_real(7)]);
    line([h2_real(8) h2_real(2)],[h2_real(7) h2_real(1)]);

    line([h2_predict(2) h2_predict(4)],[h2_predict(1) h2_predict(3)], 'color', [0,1,0]);
    line([h2_predict(4) h2_predict(6)],[h2_predict(3) h2_predict(5)], 'color', [0,1,0]);
    line([h2_predict(6) h2_predict(8)],[h2_predict(5) h2_predict(7)], 'color', [0,1,0]);
    line([h2_predict(8) h2_predict(2)],[h2_predict(7) h2_predict(1)], 'color', [0,1,0]);
    

    
    result1 = 0;
    result2 = 0;
    for j = 1 : 2 : 7
        result1 = result1 + sqrt((label(j)-predict(j))^2 + (label(j+1)-predict(j+1))^2);
        result2 = result2 + (label(j)-predict(j))^2 + (label(j+1)-predict(j+1))^2;
    end
    result1 = result1 / 4;
    result2 = result2 / 2;

    title(['Mean Corner Error = ' num2str(result1)], 'color', [1,0,0])
end