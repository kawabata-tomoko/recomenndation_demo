function showSpinner(spinnerId) {
    // 显示特定ID的旋转器
    $('#' + spinnerId).show();

    // 模拟加载时间
    setTimeout(function() {
        // 隐藏特定ID的旋转器
        $('#' + spinnerId).hide();
    }, 5000); // 这里设置为5秒，可以根据实际情况调整
}